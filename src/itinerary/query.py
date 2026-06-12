"""Itinerary query construction from sessionized check-ins.

A query asks: given the start POI, (optional) end POI, and a length budget K,
recover the user's visited sequence. The ground truth is the session itself.
See itinerary_plan.md sections 1 and 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class ItineraryQuery:
    """A single itinerary recommendation query + its ground-truth route.

    Attributes:
        user_idx: contiguous user index (same space as Phase-1 ``user_emb``).
        start_poi: POI index the route must start at (= session's first POI).
        end_poi: POI index the route must end at, or ``None`` for open-ended.
        K: length budget = number of stops to produce (= ground-truth length).
        ground_truth: the actual visited POI sequence (length K), for scoring.
    """

    user_idx: int
    start_poi: int
    end_poi: Optional[int]
    K: int
    ground_truth: List[int]


def build_eval_queries(
    test_df: pd.DataFrame,
    *,
    fixed_end: bool = True,
    min_len: int = 2,
) -> List[ItineraryQuery]:
    """Turn each test session into one itinerary query.

    Convention (PersTour / Chen 2016): the ground-truth itinerary IS the
    visited session; the query gives its start, end, and length so the decoder
    must recover the middle.

    Args:
        test_df: sessionized test DataFrame with columns
            ``session_id``, ``timestamp``, ``user_idx``, ``poi_idx``.
        fixed_end: if True, the query fixes ``end_poi`` to the session's last
            POI (the decoder reserves the final hop for it); if False,
            ``end_poi`` is None (open-ended rollout).
        min_len: skip sessions shorter than this (default 2 — a session needs
            at least a start and a next).

    Returns:
        List of :class:`ItineraryQuery`, one per qualifying session.
    """
    queries: List[ItineraryQuery] = []
    for _sid, g in test_df.groupby("session_id"):
        seq = g.sort_values("timestamp")["poi_idx"].astype(int).tolist()
        if len(seq) < min_len:
            continue
        queries.append(
            ItineraryQuery(
                user_idx=int(g["user_idx"].iloc[0]),
                start_poi=seq[0],
                end_poi=seq[-1] if fixed_end else None,
                K=len(seq),
                ground_truth=seq,
            )
        )
    return queries
