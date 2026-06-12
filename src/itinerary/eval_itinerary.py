"""Order-aware itinerary metrics (Chen 2016) + evaluation loop.

Primary metric is pairs-F1 (F1 over correctly-ordered POI pairs). Reported in a
SEPARATE table from the Phase-1 ranking metrics. See itinerary_plan.md section 6.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.itinerary.decode import encode_pois, rollout_beam, rollout_greedy
from src.itinerary.query import ItineraryQuery


def _ordered_pairs(seq: List[int]) -> Set[Tuple[int, int]]:
    """Set of ordered pairs (a, b) with a before b in the sequence.

    Args:
        seq: a route (list of POI indices).

    Returns:
        Set of (a, b) tuples for every i < j. Size = n(n-1)/2.
    """
    return {(seq[i], seq[j]) for i in range(len(seq)) for j in range(i + 1, len(seq))}


def pairs_f1(pred: List[int], truth: List[int]) -> float:
    """Order-aware pairs-F1 (Chen 2016).

    A pair (a, b) is correct iff both POIs appear in both routes and in the same
    order. pairs-F1 = harmonic mean of pairs-precision and pairs-recall.

    Args:
        pred: predicted route.
        truth: ground-truth route.

    Returns:
        pairs-F1 in [0, 1]. For routes too short to form a pair (length < 2),
        returns 1.0 iff the routes are identical, else 0.0.
    """
    pp, pt = _ordered_pairs(pred), _ordered_pairs(truth)
    if not pp or not pt:  # length < 2 on either side
        return 1.0 if list(pred) == list(truth) else 0.0
    inter = len(pp & pt)
    prec = inter / len(pp)
    rec = inter / len(pt)
    if prec + rec == 0.0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def set_f1(pred: List[int], truth: List[int]) -> float:
    """Order-agnostic F1 over the set of visited POIs.

    Args:
        pred: predicted route.
        truth: ground-truth route.

    Returns:
        set-F1 in [0, 1] ("did we pick the right places", ignoring order).
    """
    sp, st = set(pred), set(truth)
    if not sp or not st:
        return 0.0
    inter = len(sp & st)
    prec = inter / len(sp)
    rec = inter / len(st)
    if prec + rec == 0.0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def exact_match(pred: List[int], truth: List[int]) -> float:
    """1.0 if the predicted route equals the ground truth exactly, else 0.0."""
    return 1.0 if list(pred) == list(truth) else 0.0


def is_feasible(pred: List[int], K: int) -> bool:
    """True if the route is loop-free and respects the length budget K.

    Args:
        pred: predicted route.
        K: length budget.

    Returns:
        True iff no repeats and ``len(pred) <= K``.
    """
    return len(pred) == len(set(pred)) and len(pred) <= K


@torch.no_grad()
def evaluate_itinerary(
    model: nn.Module,
    queries: List[ItineraryQuery],
    edge_index: torch.Tensor,
    poi_coords: np.ndarray,
    device: torch.device,
    decoder: str = "greedy",
    beam: int = 3,
    assumed_dt_hours: float = 1.0,
    min_len: int = 1,
) -> Dict[str, float]:
    """Decode every query and average itinerary metrics.

    The GCN is encoded ONCE and shared across all queries.

    Args:
        model: trained NextPOIModel.
        queries: list of :class:`ItineraryQuery`.
        edge_index: (2, E) long graph tensor on ``device``.
        poi_coords: (|V|, 2) lat/lon array.
        device: torch device.
        decoder: ``"greedy"`` or ``"beam"``.
        beam: beam width if ``decoder == "beam"``.
        assumed_dt_hours: constant Δt per step (length mode).
        min_len: only score queries whose ground truth has at least this length
            (use 3 for the meaningful length≥3 subset).

    Returns:
        Dict with ``pairs-F1``, ``set-F1``, ``exact-match``, ``feasibility``,
        and ``n`` (number of scored queries).
    """
    model.eval()
    poi_features = encode_pois(model, edge_index)

    sums = {"pairs-F1": 0.0, "set-F1": 0.0, "exact-match": 0.0, "feasibility": 0.0}
    n = 0
    for q in queries:
        if q.K < min_len:
            continue
        if decoder == "beam":
            route = rollout_beam(
                model, q, edge_index, poi_coords, device,
                beam=beam, assumed_dt_hours=assumed_dt_hours,
                poi_features=poi_features,
            )
        else:
            route = rollout_greedy(
                model, q, edge_index, poi_coords, device,
                assumed_dt_hours=assumed_dt_hours, poi_features=poi_features,
            )
        sums["pairs-F1"] += pairs_f1(route, q.ground_truth)
        sums["set-F1"] += set_f1(route, q.ground_truth)
        sums["exact-match"] += exact_match(route, q.ground_truth)
        sums["feasibility"] += 1.0 if is_feasible(route, q.K) else 0.0
        n += 1

    out = {k: (v / n if n else 0.0) for k, v in sums.items()}
    out["n"] = float(n)
    return out


def fmt_itinerary_metrics(m: Dict[str, float], keys: Optional[Iterable[str]] = None) -> str:
    """Pretty-print an itinerary metric dict as ``key=val | ...``.

    Args:
        m: metric dict from :func:`evaluate_itinerary`.
        keys: optional ordered subset of keys; defaults to all.

    Returns:
        Single-line string for logging.
    """
    if keys is None:
        keys = m.keys()
    parts = []
    for k in keys:
        v = m[k]
        parts.append(f"{k}={int(v)}" if k == "n" else f"{k}={v:.4f}")
    return " | ".join(parts)
