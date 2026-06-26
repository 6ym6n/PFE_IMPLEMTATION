"""Strategy-B pointer-network itinerary model + its decoders and evaluator.

Architecture (see itinerary_plan.md, Strategy B):
- Encoder: reuse the Phase-1 GCN (POIGraphEncoder) for POI features H ∈ R^(|V| x d_p)
  and a user embedding e_u. The query (start, end, user) initializes the decoder.
- Decoder: a GRU rolled over the trajectory. At each step the decoder state is
  projected to d_p and scored against every POI feature by INNER PRODUCT
  (pointer attention): logits_v = <proj([h_t ; e_u]), H_v>.
- ``use_context`` (B-v2): when True, the per-step decoder input also includes the
  Δd/Δt context of the most recent hop (via the Phase-1 ContextEncoder), matching
  the information the frozen next-POI model consumes. When False, the decoder input
  is just the previous POI's GCN feature (B-v1).
- Training: teacher-forced sequence cross-entropy over whole trajectories.
- Inference: greedy / beam rollout with a no-revisit mask and reserved fixed end.

Nothing in the Phase-1 next-POI model is modified; this reuses POIGraphEncoder and
(optionally) ContextEncoder.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.data.preprocess import haversine_km
from src.itinerary.eval_itinerary import exact_match, pairs_f1, set_f1
from src.itinerary.query import ItineraryQuery
from src.models.components import ContextEncoder, POIGraphEncoder

NEG_INF = float("-inf")


class PointerItineraryModel(nn.Module):
    """Pointer-network trip recommender over a hybrid POI graph.

    Args:
        n_pois: vocabulary size |V|.
        n_users: number of users |U|.
        d_p: POI embedding / GCN dim (default 128, matches Phase-1).
        d_u: user embedding dim (default 64).
        d_h: GRU hidden dim (default 128).
        d_c: context dim used only when ``use_context`` (default 32).
        use_context: if True (B-v2), feed Δd/Δt context into the decoder input.
        n_gcn_layers: GCN depth (default 2).
        dropout: dropout in the GCN (default 0.2).
    """

    def __init__(
        self,
        n_pois: int,
        n_users: int,
        d_p: int = 128,
        d_u: int = 64,
        d_h: int = 128,
        d_c: int = 32,
        use_context: bool = False,
        n_gcn_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_context = use_context
        self.gcn = POIGraphEncoder(n_pois, d_p, n_layers=n_gcn_layers, dropout=dropout)
        self.user_emb = nn.Embedding(n_users, d_u)
        self.context = ContextEncoder(d_c) if use_context else None
        dec_in = d_p + (d_c if use_context else 0)
        self.init_proj = nn.Linear(2 * d_p + d_u, d_h)   # [H_start ; H_end ; e_u] -> h_0
        self.gru = nn.GRU(dec_in, d_h, batch_first=True)  # input = prev POI feat (+ ctx)
        self.out_proj = nn.Linear(d_h + d_u, d_p)         # decoder state -> query vector
        nn.init.xavier_uniform_(self.user_emb.weight)

    # ---- shared pieces -------------------------------------------------
    def encode_pois(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Run the GCN once. Returns (|V|, d_p) POI features."""
        return self.gcn(edge_index)

    def init_state(
        self, poi_features: torch.Tensor, start: torch.Tensor, end: torch.Tensor,
        e_u: torch.Tensor,
    ) -> torch.Tensor:
        """Build the initial GRU hidden state from the query.

        Args:
            poi_features: (|V|, d_p) GCN output.
            start: (B,) long start POI indices.
            end: (B,) long end POI indices.
            e_u: (B, d_u) user embeddings.

        Returns:
            (1, B, d_h) initial hidden state for the GRU.
        """
        z = torch.cat([poi_features[start], poi_features[end], e_u], dim=-1)
        return torch.tanh(self.init_proj(z)).unsqueeze(0)

    def _ctx_vec(self, delta_d: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """Encode one batch of scalar (Δd, Δt) into (B, d_c). Shapes are (B,)."""
        return self.context(delta_d.unsqueeze(1), delta_t.unsqueeze(1)).squeeze(1)

    def step_logits(
        self, dec_input: torch.Tensor, h: torch.Tensor, e_u: torch.Tensor,
        poi_features: torch.Tensor,
    ) -> tuple:
        """One decoder step → pointer logits over the whole vocabulary.

        Args:
            dec_input: (B, d_p [+ d_c]) decoder input (prev POI feat, optional ctx).
            h: (1, B, d_h) current GRU hidden state.
            e_u: (B, d_u) user embeddings.
            poi_features: (|V|, d_p) GCN output.

        Returns:
            ``(logits, h_next)`` where logits is (B, |V|) and h_next is (1, B, d_h).
        """
        out, h_next = self.gru(dec_input.unsqueeze(1), h)    # out: (B, 1, d_h)
        q = self.out_proj(torch.cat([out.squeeze(1), e_u], dim=-1))  # (B, d_p)
        logits = q @ poi_features.t()                        # (B, |V|) inner-product pointer
        return logits, h_next

    # ---- training forward (teacher forced) -----------------------------
    def forward(
        self, poi_seq: torch.Tensor, lengths: torch.Tensor, user_ids: torch.Tensor,
        edge_index: torch.Tensor,
        delta_d: Optional[torch.Tensor] = None, delta_t: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Teacher-forced forward over whole trajectories.

        Predicts steps 1..T-1 given the ground-truth prefix at each step. When
        ``use_context``, the decoder input at step t also carries the Δd/Δt of the
        hop that arrived at position t-1 (delta_*[:, t-1]).

        Args:
            poi_seq: (B, T) long padded ordered sequences (pad value 0).
            lengths: (B,) long true sequence lengths.
            user_ids: (B,) long.
            edge_index: (2, E) long graph on the model device.
            delta_d, delta_t: (B, T) float per-step gaps (required if use_context).

        Returns:
            ``(logits, targets, mask)``:
                - logits  (B, T-1, |V|) pointer logits per step
                - targets (B, T-1)      the next POI at each step
                - mask    (B, T-1) bool  True where the step is real (not padding)
        """
        B, T = poi_seq.shape
        device = poi_seq.device
        H = self.encode_pois(edge_index)
        e_u = self.user_emb(user_ids)
        start = poi_seq[:, 0]
        end = poi_seq[torch.arange(B, device=device), lengths - 1]
        h = self.init_state(H, start, end, e_u)

        ctx = None
        if self.use_context:
            if delta_d is None or delta_t is None:
                raise ValueError("use_context=True requires delta_d and delta_t")
            ctx = self.context(delta_d, delta_t)             # (B, T, d_c)

        step_logits = []
        for t in range(1, T):
            dec_input = H[poi_seq[:, t - 1]]                 # (B, d_p) teacher forcing
            if self.use_context:
                dec_input = torch.cat([dec_input, ctx[:, t - 1, :]], dim=-1)
            logits_t, h = self.step_logits(dec_input, h, e_u, H)
            step_logits.append(logits_t)
        logits = torch.stack(step_logits, dim=1)             # (B, T-1, |V|)
        targets = poi_seq[:, 1:]                             # (B, T-1)
        steps = torch.arange(1, T, device=device).unsqueeze(0)      # (1, T-1)
        mask = steps < lengths.unsqueeze(1)                          # (B, T-1)
        return logits, targets, mask


# ---------------------------------------------------------------------------
# Inference decoders (mirror Strategy A's discipline: loop-free + reserved end)
# ---------------------------------------------------------------------------
def _decode_ctx_vec(
    model: PointerItineraryModel, route: List[int], step: int,
    poi_coords: Optional[np.ndarray], assumed_dt_hours: float, device: torch.device,
) -> Optional[torch.Tensor]:
    """Context vector for the current position during decoding, or None.

    The current position is route[-1] (= position step-1). Its context is the hop
    route[-2] -> route[-1]: Δd from coordinates, Δt = the assumed constant. The
    first position (step == 1) has Δd = Δt = 0 by convention (matches Strategy A).
    """
    if not model.use_context:
        return None
    if step == 1 or poi_coords is None:
        dd, dt = 0.0, 0.0
    else:
        a, b = route[-2], route[-1]
        dd = min(float(haversine_km(poi_coords[a, 0], poi_coords[a, 1],
                                    poi_coords[b, 0], poi_coords[b, 1])), 100.0)
        dt = min(max(assumed_dt_hours, 0.0), 24.0)
    return model._ctx_vec(
        torch.tensor([dd], dtype=torch.float, device=device),
        torch.tensor([dt], dtype=torch.float, device=device),
    )


@torch.no_grad()
def pointer_rollout_greedy(
    model: PointerItineraryModel,
    query: ItineraryQuery,
    edge_index: torch.Tensor,
    device: torch.device,
    poi_features: Optional[torch.Tensor] = None,
    poi_coords: Optional[np.ndarray] = None,
    assumed_dt_hours: float = 1.0,
) -> List[int]:
    """Greedy decode from the pointer model (loop-free, reserved fixed end).

    Args:
        model: trained :class:`PointerItineraryModel` (eval mode set internally).
        query: itinerary query (start, end, K, user).
        edge_index: (2, E) long graph on ``device``.
        device: torch device.
        poi_features: optional cached GCN output; computed if None.
        poi_coords: (|V|, 2) lat/lon array — required if model.use_context.
        assumed_dt_hours: constant Δt used for the context features (use_context only).

    Returns:
        Ordered route of length ``query.K``, loop-free, ending at the fixed end.
    """
    model.eval()
    H = model.encode_pois(edge_index) if poi_features is None else poi_features
    e_u = model.user_emb(torch.tensor([query.user_idx], device=device))
    end_for_init = query.end_poi if query.end_poi is not None else query.start_poi
    h = model.init_state(H, torch.tensor([query.start_poi], device=device),
                         torch.tensor([end_for_init], device=device), e_u)

    route = [query.start_poi]
    visited = {query.start_poi}
    for step in range(1, query.K):
        is_last = step == query.K - 1
        dec_input = H[torch.tensor([route[-1]], device=device)]
        ctx_vec = _decode_ctx_vec(model, route, step, poi_coords, assumed_dt_hours, device)
        if ctx_vec is not None:
            dec_input = torch.cat([dec_input, ctx_vec], dim=-1)
        logits, h = model.step_logits(dec_input, h, e_u, H)
        logits = logits.squeeze(0)
        if is_last and query.end_poi is not None and query.end_poi not in visited:
            nxt = query.end_poi
        else:
            block = set(visited)
            if query.end_poi is not None and not is_last:
                block.add(query.end_poi)
            logits = logits.clone()
            if block:
                logits[list(block)] = NEG_INF
            nxt = int(torch.argmax(logits).item())
        route.append(nxt)
        visited.add(nxt)
    return route


@torch.no_grad()
def pointer_rollout_beam(
    model: PointerItineraryModel,
    query: ItineraryQuery,
    edge_index: torch.Tensor,
    device: torch.device,
    beam: int = 3,
    poi_features: Optional[torch.Tensor] = None,
    poi_coords: Optional[np.ndarray] = None,
    assumed_dt_hours: float = 1.0,
) -> List[int]:
    """Beam decode from the pointer model (``beam=1`` == greedy).

    Args:
        model: trained PointerItineraryModel.
        query: itinerary query.
        edge_index: (2, E) long graph on ``device``.
        device: torch device.
        beam: beam width (≥ 1).
        poi_features: optional cached GCN output.
        poi_coords: (|V|, 2) lat/lon array — required if model.use_context.
        assumed_dt_hours: constant Δt used for the context features.

    Returns:
        Ordered route of length ``query.K``, loop-free, ending at the fixed end.
    """
    model.eval()
    H = model.encode_pois(edge_index) if poi_features is None else poi_features
    e_u = model.user_emb(torch.tensor([query.user_idx], device=device))
    end_for_init = query.end_poi if query.end_poi is not None else query.start_poi
    h0 = model.init_state(H, torch.tensor([query.start_poi], device=device),
                          torch.tensor([end_for_init], device=device), e_u)
    if query.K <= 1:
        return [query.start_poi]

    beams = [([query.start_poi], 0.0, {query.start_poi}, h0)]
    for step in range(1, query.K):
        is_last = step == query.K - 1
        candidates = []
        for route, score, visited, h in beams:
            dec_input = H[torch.tensor([route[-1]], device=device)]
            ctx_vec = _decode_ctx_vec(model, route, step, poi_coords, assumed_dt_hours, device)
            if ctx_vec is not None:
                dec_input = torch.cat([dec_input, ctx_vec], dim=-1)
            logits, h_next = model.step_logits(dec_input, h, e_u, H)
            logp = torch.log_softmax(logits.squeeze(0), dim=-1)
            if is_last and query.end_poi is not None and query.end_poi not in visited:
                nxt = query.end_poi
                candidates.append(
                    (route + [nxt], score + float(logp[nxt].item()), visited | {nxt}, h_next)
                )
                continue
            masked = logp.clone()
            block = set(visited)
            if query.end_poi is not None and not is_last:
                block.add(query.end_poi)
            if block:
                masked[list(block)] = NEG_INF
            k = min(beam, masked.numel())
            for c in torch.topk(masked, k).indices.tolist():
                val = float(masked[c].item())
                if val == NEG_INF:
                    continue
                candidates.append((route + [c], score + val, visited | {c}, h_next))
        if not candidates:
            break
        candidates.sort(key=lambda t: t[1], reverse=True)
        beams = candidates[:beam]
    return beams[0][0]


@torch.no_grad()
def evaluate_pointer(
    model: PointerItineraryModel,
    queries: List[ItineraryQuery],
    edge_index: torch.Tensor,
    device: torch.device,
    decoder: str = "greedy",
    beam: int = 3,
    min_len: int = 1,
    poi_coords: Optional[np.ndarray] = None,
    assumed_dt_hours: float = 1.0,
) -> Dict[str, float]:
    """Decode queries with the pointer model and average itinerary metrics.

    Args:
        model: trained PointerItineraryModel.
        queries: list of :class:`ItineraryQuery`.
        edge_index: (2, E) long graph on ``device``.
        device: torch device.
        decoder: ``"greedy"`` or ``"beam"``.
        beam: beam width if decoder == "beam".
        min_len: only score queries with ground-truth length ≥ this.
        poi_coords: (|V|, 2) lat/lon — required if model.use_context.
        assumed_dt_hours: constant Δt for context features (use_context only).

    Returns:
        Dict with ``pairs-F1``, ``set-F1``, ``exact-match``, ``feasibility``, ``n``.
    """
    model.eval()
    H = model.encode_pois(edge_index)
    sums = {"pairs-F1": 0.0, "set-F1": 0.0, "exact-match": 0.0, "feasibility": 0.0}
    n = 0
    for q in queries:
        if q.K < min_len:
            continue
        if decoder == "beam":
            r = pointer_rollout_beam(model, q, edge_index, device, beam=beam,
                                     poi_features=H, poi_coords=poi_coords,
                                     assumed_dt_hours=assumed_dt_hours)
        else:
            r = pointer_rollout_greedy(model, q, edge_index, device, poi_features=H,
                                       poi_coords=poi_coords,
                                       assumed_dt_hours=assumed_dt_hours)
        sums["pairs-F1"] += pairs_f1(r, q.ground_truth)
        sums["set-F1"] += set_f1(r, q.ground_truth)
        sums["exact-match"] += exact_match(r, q.ground_truth)
        sums["feasibility"] += 1.0 if (len(r) == len(set(r)) and len(r) <= q.K) else 0.0
        n += 1
    out = {k: (v / n if n else 0.0) for k, v in sums.items()}
    out["n"] = float(n)
    return out
