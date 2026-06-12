"""Inference-only itinerary decoders over the frozen next-POI model.

An itinerary is produced by rolling out the Phase-1 ``NextPOIModel`` next-POI
scorer under a no-revisit mask and a length budget, reserving the final hop for
a fixed end POI. The trained model is NOT modified. See itinerary_plan.md.

Performance: the GCN output is input-independent, so POI features are computed
ONCE (``model.gcn(edge_index)``) and reused across every decode step / query.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.data.preprocess import haversine_km
from src.itinerary.query import ItineraryQuery

NEG_INF = float("-inf")


def encode_pois(model: nn.Module, edge_index: torch.Tensor) -> torch.Tensor:
    """Run the GCN once to get the full POI feature matrix.

    Args:
        model: a trained ``NextPOIModel`` (uses its ``.gcn`` submodule).
        edge_index: (2, E) long tensor of the hybrid graph, on the model device.

    Returns:
        (|V|, d_p) POI feature tensor (cache and reuse across decode steps).
    """
    return model.gcn(edge_index)


def _step_logits(
    model: nn.Module,
    poi_features: torch.Tensor,
    route: List[int],
    user_idx: int,
    poi_coords: np.ndarray,
    device: torch.device,
    assumed_dt_hours: float,
) -> torch.Tensor:
    """Score the next POI given the current partial route (GCN reused).

    Replicates ``NextPOIModel.forward`` for a single-example batch but consumes
    a precomputed ``poi_features`` instead of re-running the GCN.

    Args:
        model: trained NextPOIModel (uses ``.context``, ``.gru``, ``.user_emb``, ``.head``).
        poi_features: (|V|, d_p) cached GCN output.
        route: current partial route (list of POI indices, length T ≥ 1).
        user_idx: user index for the query.
        poi_coords: (|V|, 2) array of [lat, lon] degrees, for Δd.
        device: torch device.
        assumed_dt_hours: constant Δt fed per step (length mode; see design doc).

    Returns:
        (|V|,) tensor of next-POI logits.
    """
    T = len(route)
    poi_ids = torch.tensor([route], dtype=torch.long, device=device)  # (1, T)

    dd = [0.0]
    dt = [0.0]
    for i in range(1, T):
        a, b = route[i - 1], route[i]
        dist = float(
            haversine_km(
                poi_coords[a, 0], poi_coords[a, 1],
                poi_coords[b, 0], poi_coords[b, 1],
            )
        )
        dd.append(min(dist, 100.0))                 # clip to match preprocess
        dt.append(min(max(assumed_dt_hours, 0.0), 24.0))
    delta_d = torch.tensor([dd], dtype=torch.float, device=device)   # (1, T)
    delta_t = torch.tensor([dt], dtype=torch.float, device=device)   # (1, T)
    user_ids = torch.tensor([user_idx], dtype=torch.long, device=device)
    lengths = torch.tensor([T], dtype=torch.long)                    # CPU for packing

    seq = poi_features[poi_ids]                      # (1, T, d_p)
    ctx = model.context(delta_d, delta_t)            # (1, T, d_c)
    x = torch.cat([seq, ctx], dim=-1)                # (1, T, d_p + d_c)
    packed = nn.utils.rnn.pack_padded_sequence(
        x, lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    _, h_T = model.gru(packed)                       # (1, 1, d_h)
    h_T = h_T.squeeze(0)                             # (1, d_h)
    u = model.user_emb(user_ids)                     # (1, d_u)
    z = torch.cat([h_T, u], dim=-1)                  # (1, d_h + d_u)
    return model.head(z).squeeze(0)                  # (|V|,)


@torch.no_grad()
def rollout_greedy(
    model: nn.Module,
    query: ItineraryQuery,
    edge_index: torch.Tensor,
    poi_coords: np.ndarray,
    device: torch.device,
    assumed_dt_hours: float = 1.0,
    poi_features: Optional[torch.Tensor] = None,
) -> List[int]:
    """Greedy budget-constrained, loop-free rollout (the baseline floor).

    Starts at ``query.start_poi`` and repeatedly appends the argmax non-visited
    POI until the length budget ``K`` is reached. The final stop is forced to
    ``query.end_poi`` when fixed (reserve-the-return-leg).

    Args:
        model: trained NextPOIModel (eval mode recommended).
        query: the itinerary query.
        edge_index: (2, E) long graph tensor on ``device``.
        poi_coords: (|V|, 2) lat/lon array.
        device: torch device.
        assumed_dt_hours: constant Δt per step (length mode).
        poi_features: optional precomputed GCN output; computed if None.

    Returns:
        Ordered list of POI indices, length == ``query.K``, no repeats.
    """
    model.eval()
    if poi_features is None:
        poi_features = encode_pois(model, edge_index)

    route = [query.start_poi]
    visited = {query.start_poi}
    for step in range(1, query.K):
        is_last = step == query.K - 1
        if is_last and query.end_poi is not None and query.end_poi not in visited:
            nxt = query.end_poi
        else:
            logits = _step_logits(
                model, poi_features, route, query.user_idx,
                poi_coords, device, assumed_dt_hours,
            ).clone()
            # Block already-visited POIs (loop-free) and, at non-final steps,
            # the fixed end POI — it is reserved for the last position so it
            # cannot be consumed early (reserve-the-return-leg).
            block = set(visited)
            if query.end_poi is not None and not is_last:
                block.add(query.end_poi)
            if block:
                logits[list(block)] = NEG_INF
            nxt = int(torch.argmax(logits).item())
        route.append(nxt)
        visited.add(nxt)
    return route


@torch.no_grad()
def rollout_beam(
    model: nn.Module,
    query: ItineraryQuery,
    edge_index: torch.Tensor,
    poi_coords: np.ndarray,
    device: torch.device,
    beam: int = 3,
    assumed_dt_hours: float = 1.0,
    poi_features: Optional[torch.Tensor] = None,
) -> List[int]:
    """Beam-search rollout (less myopic than greedy; ``beam=1`` == greedy).

    Maintains the top-``beam`` partial routes ranked by summed transition
    log-probability and returns the highest-scoring complete route. The final
    stop is forced to ``query.end_poi`` when fixed.

    Args:
        model: trained NextPOIModel.
        query: the itinerary query.
        edge_index: (2, E) long graph tensor on ``device``.
        poi_coords: (|V|, 2) lat/lon array.
        device: torch device.
        beam: beam width (≥ 1).
        assumed_dt_hours: constant Δt per step (length mode).
        poi_features: optional precomputed GCN output; computed if None.

    Returns:
        Ordered list of POI indices, length == ``query.K``, no repeats.
    """
    model.eval()
    if poi_features is None:
        poi_features = encode_pois(model, edge_index)

    if query.K <= 1:
        return [query.start_poi]

    # Each beam entry: (route, cumulative_logprob, visited_set)
    beams = [([query.start_poi], 0.0, {query.start_poi})]
    for step in range(1, query.K):
        is_last = step == query.K - 1
        candidates = []
        for route, score, visited in beams:
            logits = _step_logits(
                model, poi_features, route, query.user_idx,
                poi_coords, device, assumed_dt_hours,
            )
            logp = torch.log_softmax(logits, dim=-1)

            if is_last and query.end_poi is not None and query.end_poi not in visited:
                nxt = query.end_poi
                candidates.append(
                    (route + [nxt], score + float(logp[nxt].item()), visited | {nxt})
                )
                continue

            # Block visited (loop-free) and, at non-final steps, the reserved
            # fixed end POI so it is only placed at the last position.
            masked = logp.clone()
            block = set(visited)
            if query.end_poi is not None and not is_last:
                block.add(query.end_poi)
            if block:
                masked[list(block)] = NEG_INF
            k = min(beam, masked.numel())
            top = torch.topk(masked, k).indices.tolist()
            for c in top:
                val = float(masked[c].item())
                if val == NEG_INF:
                    continue
                candidates.append((route + [c], score + val, visited | {c}))

        if not candidates:  # all POIs exhausted (|V| < K); stop early
            break
        candidates.sort(key=lambda t: t[1], reverse=True)
        beams = candidates[:beam]

    return beams[0][0]
