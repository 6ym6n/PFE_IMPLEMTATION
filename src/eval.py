"""Ranking metrics for next-POI evaluation: HR@k, NDCG@k, MRR.

Mirrors Section 8/9 of implementation_guide.md exactly.

Ranks are computed as ``(logits > target_score).sum() + 1`` over the full
vocabulary — no negative sampling, no candidate restriction. This matches
the GETNext / STHGCN evaluation convention used by the LLM4POI table.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute HR@k, NDCG@k, MRR averaged over all examples in ``loader``.

    HR@k is 1 if the target rank ≤ k, else 0.
    NDCG@k = HR@k / log2(rank + 1).
    MRR    = 1 / rank.

    Args:
        model: trained :class:`NextPOIModel` (or compatible) in any mode;
            this function calls ``.eval()``.
        loader: DataLoader yielding the standard collated batch dict.
        edge_index: (2, E) long tensor on ``device``.
        device: torch device for batch tensors.
        ks: tuple of cutoffs (default (1, 5, 10)).

    Returns:
        Dict mapping metric name → float (e.g. ``{"HR@1": ..., "NDCG@5": ..., "MRR": ...}``).
    """
    model.eval()
    sums: Dict[str, float] = {f"HR@{k}": 0.0 for k in ks}
    sums.update({f"NDCG@{k}": 0.0 for k in ks})
    sums["MRR"] = 0.0
    n = 0

    for batch in loader:
        poi_ids = batch["poi_ids"].to(device)
        delta_d = batch["delta_d"].to(device)
        delta_t = batch["delta_t"].to(device)
        user_ids = batch["user_ids"].to(device)
        lengths = batch["lengths"]
        targets = batch["targets"].to(device)

        logits = model(poi_ids, delta_d, delta_t, user_ids, lengths, edge_index)

        target_scores = logits.gather(1, targets.unsqueeze(1))   # (B, 1)
        ranks = (logits > target_scores).sum(dim=1) + 1          # (B,)
        ranks_f = ranks.float()

        for k in ks:
            hit = (ranks <= k).float()
            sums[f"HR@{k}"] += hit.sum().item()
            sums[f"NDCG@{k}"] += (hit / torch.log2(ranks_f + 1)).sum().item()
        sums["MRR"] += (1.0 / ranks_f).sum().item()
        n += int(targets.size(0))

    return {k: v / n for k, v in sums.items()}


def fmt_metrics(m: Dict[str, float], keys: Iterable[str] | None = None) -> str:
    """Pretty-print metric dict as ``key=val | key=val | ...``.

    Args:
        m: metric dict from :func:`evaluate_model`.
        keys: optional iterable of keys to include in order. Defaults to all.

    Returns:
        Single-line string suitable for logging.
    """
    if keys is None:
        keys = m.keys()
    return " | ".join(f"{k}={m[k]:.4f}" for k in keys)
