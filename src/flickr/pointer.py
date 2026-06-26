"""Light GCN + pointer recommender for the small Flickr datasets (Colab).

The learned recommender for Strategy D. It trims the Foursquare pointer model
(``src.itinerary.pointer_model``) to the tiny Flickr vocabularies (27-88 POIs)
and the leave-one-out protocol, where a *fresh* model is trained on every fold:

* **Encoder.** A 2-layer GCN over a per-fold POI graph (geographic kNN union
  training-fold co-visit transitions) produces POI features ``H`` (|V| x d).
  Optional user embedding (off by default — the leave-one-out folds give almost
  no per-user signal on these datasets, matching the literature which is not
  user-personalised here).
* **Decoder.** A GRU seeded from ``tanh(W[H_start ; H_end])`` (the query), rolled
  over the trajectory; at each step the state is scored against every POI by
  inner product (a pointer), masked to be loop-free with the end reserved for the
  final hop — the same discipline as the Foursquare decoders.
* **Training.** Teacher-forced sequence cross-entropy over whole training-fold
  trajectories (length >= 3). Small + fast: each fold trains in well under a
  second on a GPU, so strict leave-one-out is tractable.

This module imports torch lazily-friendly (top-level import is fine on Colab and
locally, where torch-CPU is installed) and depends only on numpy + torch +
:mod:`src.flickr.data`. No global state; every hyperparameter is an argument.

NOTE: training is intended for **Colab** (see ``colab_itinerary.ipynb``). The
unit tests exercise only forward/decoder *shape & invariants* on a tiny random
model (CPU, no real training), per the project convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.flickr.data import FlickrCity, Trajectory
from src.itinerary.query import ItineraryQuery

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Per-fold POI graph (geographic kNN ∪ training co-visit), self-contained
# ---------------------------------------------------------------------------
def build_poi_graph(
    city: FlickrCity,
    train: Sequence[Trajectory],
    *,
    knn_k: int = 6,
    covisit_threshold: int = 1,
) -> torch.Tensor:
    """Build a symmetric PoI ``edge_index`` from kNN ∪ training co-visit edges.

    Uses a plain haversine kNN (no sklearn dependency — the vocab is tiny) so the
    graph always connects every POI, unioned with the consecutive-transition
    co-visit edges from the *training fold only* (no leakage from the held-out
    trajectory).

    Args:
        city: the city (for coordinates and |V|).
        train: training-fold trajectories (co-visit edges come from these).
        knn_k: geographic neighbours per POI.
        covisit_threshold: minimum co-visit count to add a transition edge.

    Returns:
        (2, E) long tensor, each undirected edge present in both directions,
        with self-loops on every node (so isolated POIs still self-update).
    """
    n = city.n_pois
    coords = np.radians(city.poi_coords)  # (n, 2) lat, lon in radians
    # pairwise haversine via broadcasting (n is small)
    lat = coords[:, 0:1]
    lon = coords[:, 1:2]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    dist = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))  # (n, n) great-circle angle

    edges: set = set()
    k = min(knn_k, n - 1) if n > 1 else 0
    for i in range(n):
        if k > 0:
            nn_idx = np.argsort(dist[i])[1 : k + 1]  # skip self at 0
            for j in nn_idx:
                a_, b_ = (i, int(j)) if i < int(j) else (int(j), i)
                edges.add((a_, b_))

    counts: dict = {}
    for t in train:
        seq = t.pois
        for u in range(len(seq) - 1):
            a_, b_ = seq[u], seq[u + 1]
            if a_ == b_:
                continue
            key = (min(a_, b_), max(a_, b_))
            counts[key] = counts.get(key, 0) + 1
    for key, c in counts.items():
        if c >= covisit_threshold:
            edges.add(key)

    src: List[int] = list(range(n))   # self-loops
    dst: List[int] = list(range(n))
    for a_, b_ in edges:
        src.extend([a_, b_])
        dst.extend([b_, a_])
    return torch.tensor([src, dst], dtype=torch.long)


# ---------------------------------------------------------------------------
# Minimal 2-layer GCN (no torch_geometric dependency — runs anywhere)
# ---------------------------------------------------------------------------
class _GCN(nn.Module):
    """Tiny symmetric-normalised 2-layer GCN over a learnable POI embedding table.

    Self-contained (does not need torch_geometric), since the Flickr graphs are
    small. Computes ``Â = D^-1/2 (A) D^-1/2`` once per forward from ``edge_index``.

    Args:
        n_pois: vocabulary size |V|.
        dim: embedding / hidden dim d.
        dropout: dropout between the two layers.
    """

    def __init__(self, n_pois: int, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_pois, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, edge_index: torch.Tensor, n_pois: int) -> torch.Tensor:
        """Return (|V|, d) POI features after 2 GCN layers.

        Args:
            edge_index: (2, E) long tensor (already symmetric, with self-loops).
            n_pois: |V| (for the normalisation denominator).

        Returns:
            (|V|, d) float tensor of POI features.
        """
        device = self.emb.weight.device
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(n_pois, device=device).index_add_(
            0, src, torch.ones_like(src, dtype=torch.float)
        )
        dinv = deg.clamp(min=1.0).pow(-0.5)
        norm = dinv[src] * dinv[dst]  # (E,)

        def propagate(x: torch.Tensor) -> torch.Tensor:
            msg = x[src] * norm.unsqueeze(-1)
            out = torch.zeros_like(x).index_add_(0, dst, msg)
            return out

        h = self.emb.weight
        h = F.relu(self.lin1(propagate(h)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(propagate(h))
        return h


# ---------------------------------------------------------------------------
# Pointer recommender
# ---------------------------------------------------------------------------
class FlickrPointerNet(nn.Module):
    """GCN-encoder + GRU pointer-decoder trip recommender for small vocabularies.

    Args:
        n_pois: vocabulary size |V|.
        dim: GCN / GRU hidden dim (default 64; the data is tiny).
        dropout: GCN dropout.
    """

    def __init__(self, n_pois: int, dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.n_pois = n_pois
        self.dim = dim
        self.gcn = _GCN(n_pois, dim, dropout=dropout)
        self.init_proj = nn.Linear(2 * dim, dim)   # [H_start ; H_end] -> h0
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.out_proj = nn.Linear(dim, dim)         # decoder state -> query vec

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Run the GCN once; returns (|V|, dim) POI features."""
        return self.gcn(edge_index, self.n_pois)

    def init_state(self, H: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """Initial GRU hidden state from the query endpoints.

        Args:
            H: (|V|, dim) POI features.
            start: (B,) long start POI indices.
            end: (B,) long end POI indices.

        Returns:
            (1, B, dim) initial hidden state.
        """
        z = torch.cat([H[start], H[end]], dim=-1)
        return torch.tanh(self.init_proj(z)).unsqueeze(0)

    def step(self, prev_feat: torch.Tensor, h: torch.Tensor, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One decoder step -> pointer logits over the vocabulary.

        Args:
            prev_feat: (B, dim) previous POI's GCN feature.
            h: (1, B, dim) GRU hidden state.
            H: (|V|, dim) POI features.

        Returns:
            ``(logits, h_next)``: logits (B, |V|), h_next (1, B, dim).
        """
        out, h_next = self.gru(prev_feat.unsqueeze(1), h)   # (B, 1, dim)
        q = self.out_proj(out.squeeze(1))                   # (B, dim)
        logits = q @ H.t()                                  # (B, |V|)
        return logits, h_next

    def forward(
        self, poi_seq: torch.Tensor, lengths: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced forward over whole padded trajectories.

        Args:
            poi_seq: (B, T) long padded ordered POI sequences (pad value 0).
            lengths: (B,) long true lengths.
            edge_index: (2, E) long graph on the model device.

        Returns:
            ``(logits, targets, mask)``:
                logits (B, T-1, |V|), targets (B, T-1), mask (B, T-1) bool.
        """
        B, T = poi_seq.shape
        device = poi_seq.device
        H = self.encode(edge_index)
        start = poi_seq[:, 0]
        end = poi_seq[torch.arange(B, device=device), lengths - 1]
        h = self.init_state(H, start, end)
        outs = []
        for t in range(1, T):
            logits_t, h = self.step(H[poi_seq[:, t - 1]], h, H)
            outs.append(logits_t)
        logits = torch.stack(outs, dim=1)                   # (B, T-1, |V|)
        targets = poi_seq[:, 1:]
        steps = torch.arange(1, T, device=device).unsqueeze(0)
        mask = steps < lengths.unsqueeze(1)
        return logits, targets, mask


# ---------------------------------------------------------------------------
# Decoding (loop-free, reserved fixed end) -> route for a query
# ---------------------------------------------------------------------------
@torch.no_grad()
def pointer_decode(
    model: FlickrPointerNet,
    query: ItineraryQuery,
    edge_index: torch.Tensor,
    device: torch.device,
    *,
    beam: int = 1,
    H: Optional[torch.Tensor] = None,
) -> List[int]:
    """Greedy (``beam==1``) or beam decode of a loop-free route for one query.

    Args:
        model: a :class:`FlickrPointerNet` (eval mode set internally).
        query: the itinerary query (start, end, K).
        edge_index: (2, E) long graph on ``device``.
        device: torch device.
        beam: beam width (1 == greedy).
        H: optional cached GCN features (shared across queries in a fold).

    Returns:
        Ordered loop-free route of length ``query.K`` from start to fixed end.
    """
    model.eval()
    if H is None:
        H = model.encode(edge_index)
    start, end, K = query.start_poi, query.end_poi, query.K
    end_for_init = end if end is not None else start
    h0 = model.init_state(
        H,
        torch.tensor([start], device=device),
        torch.tensor([end_for_init], device=device),
    )
    if K <= 1:
        return [start]

    beams = [([start], 0.0, frozenset({start}), h0)]
    for step in range(1, K):
        is_last = step == K - 1
        cand = []
        for route, score, visited, h in beams:
            logits, h_next = model.step(H[torch.tensor([route[-1]], device=device)], h, H)
            logp = torch.log_softmax(logits.squeeze(0), dim=-1)
            if is_last and end is not None and end not in visited:
                cand.append((route + [end], score + float(logp[end]), visited | {end}, h_next))
                continue
            masked = logp.clone()
            block = list(visited)
            if end is not None and not is_last:
                block.append(end)
            masked[block] = NEG_INF
            kk = min(beam, masked.numel())
            for j in torch.topk(masked, kk).indices.tolist():
                v = float(masked[j])
                if v == NEG_INF:
                    continue
                cand.append((route + [j], score + v, visited | {j}, h_next))
        if not cand:
            break
        cand.sort(key=lambda x: x[1], reverse=True)
        beams = cand[:beam]
    return beams[0][0]


# ---------------------------------------------------------------------------
# Training (Colab) — one fold
# ---------------------------------------------------------------------------
@dataclass
class PointerConfig:
    """Hyperparameters for training :class:`FlickrPointerNet` on one fold.

    Attributes:
        dim: hidden dim. epochs: training epochs. lr: Adam learning rate.
        weight_decay: Adam weight decay. dropout: GCN dropout.
        knn_k: geographic neighbours in the POI graph.
        covisit_threshold: min co-visit count for a transition edge.
        beam: decode beam width at eval. seed: torch seed for this fold.
    """

    dim: int = 64
    epochs: int = 40
    lr: float = 5e-3
    weight_decay: float = 1e-5
    dropout: float = 0.2
    knn_k: int = 6
    covisit_threshold: int = 1
    beam: int = 3
    seed: int = 0


def _pad_batch(trajs: Sequence[Trajectory], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a set of trajectories into ``(poi_seq, lengths)`` tensors.

    Args:
        trajs: trajectories (each length >= 2).
        device: torch device.

    Returns:
        ``(poi_seq, lengths)``: poi_seq (B, T_max) long padded with 0, lengths (B,).
    """
    lengths = [len(t.pois) for t in trajs]
    T = max(lengths)
    B = len(trajs)
    poi_seq = torch.zeros(B, T, dtype=torch.long)
    for i, t in enumerate(trajs):
        poi_seq[i, : len(t.pois)] = torch.tensor(t.pois, dtype=torch.long)
    return poi_seq.to(device), torch.tensor(lengths, dtype=torch.long, device=device)


def train_pointer_fold(
    city: FlickrCity,
    train: Sequence[Trajectory],
    device: torch.device,
    config: PointerConfig,
    *,
    min_len: int = 3,
) -> Tuple[FlickrPointerNet, torch.Tensor]:
    """Train a fresh pointer model on one fold's training trajectories.

    Args:
        city: the city (for |V| and coordinates).
        train: training-fold trajectories (the held-out one already removed).
        device: torch device.
        config: :class:`PointerConfig` hyperparameters.
        min_len: only train on trajectories of at least this length (default 3;
            length-2 trajectories carry a single forced [start, end] step and add
            little, but can be included by setting min_len=2).

    Returns:
        ``(model, edge_index)`` — the trained model (eval mode) and the per-fold
        graph it was trained with (reuse for decoding).
    """
    torch.manual_seed(config.seed)
    train_trajs = [t for t in train if len(t.pois) >= min_len]
    # Build the graph from the SAME (length>=min_len) pool used for the loss, so
    # the neural fold sees exactly the co-visit structure the comparable protocol
    # exposes (not the length<3 transitions that a raw `train` would smuggle in).
    edge_index = build_poi_graph(
        city, train_trajs, knn_k=config.knn_k, covisit_threshold=config.covisit_threshold
    ).to(device)
    model = FlickrPointerNet(city.n_pois, dim=config.dim, dropout=config.dropout).to(device)
    if not train_trajs:  # degenerate fold (no other length>=min_len traj); cannot train
        model.eval()
        return model, edge_index

    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    poi_seq, lengths = _pad_batch(train_trajs, device)
    model.train()
    for _ in range(config.epochs):
        opt.zero_grad()
        logits, targets, mask = model(poi_seq, lengths, edge_index)
        loss = F.cross_entropy(
            logits[mask], targets[mask], reduction="mean"
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    model.eval()
    return model, edge_index


def pointer_factory(device: torch.device, config: Optional[PointerConfig] = None, *, min_len: int = 3):
    """Return a :data:`~src.flickr.evaluate.RecommenderFactory` for the pointer net.

    Each call trains a fresh model on the fold (leave-one-out), then wraps it in a
    recommender that decodes with the cached per-fold graph + GCN features.

    Args:
        device: torch device to train/decode on.
        config: :class:`PointerConfig` (defaults if None).
        min_len: minimum training trajectory length (default 3).

    Returns:
        A factory ``factory(city, train) -> recommender``.
    """
    cfg = config or PointerConfig()

    def factory(city: FlickrCity, train: List[Trajectory]):
        # Degenerate fold (no other length>=min_len trajectory to train on):
        # fall back to a defined popularity route rather than untrained-net noise.
        # Unreachable on the real Flickr data (every city has >=47 length>=3 trajs).
        if not any(len(t.pois) >= min_len for t in train):
            from src.flickr.baselines import PopularityRecommender, _visit_counts
            return PopularityRecommender(_visit_counts(city.n_pois, train))

        model, edge_index = train_pointer_fold(city, train, device, cfg, min_len=min_len)
        H = model.encode(edge_index)

        class _Rec:
            def recommend(self, query: ItineraryQuery) -> List[int]:
                return pointer_decode(model, query, edge_index, device, beam=cfg.beam, H=H)

        return _Rec()

    return factory
