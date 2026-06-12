"""Tests for the itinerary layer: pairs-F1 correctness + decode invariants.

The pairs-F1 cases are hand-computed (the design doc requires the metric be
pinned before any numbers are generated). Decode tests use a tiny randomly
initialized NextPOIModel — outputs are arbitrary but the structural invariants
(loop-free, length == K, fixed end last, greedy == beam(1)) must always hold.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.itinerary.decode import rollout_beam, rollout_greedy
from src.itinerary.eval_itinerary import (
    evaluate_itinerary,
    exact_match,
    is_feasible,
    pairs_f1,
    set_f1,
)
from src.itinerary.query import ItineraryQuery, build_eval_queries
from src.models.next_poi import NextPOIModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# pairs-F1 — hand-checked worked examples
# --------------------------------------------------------------------------
def test_pairs_f1_perfect_match():
    assert pairs_f1([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0


def test_pairs_f1_reversed_middle():
    # truth [s,a,b,e]=[0,1,2,3], pred swaps a,b -> [0,2,1,3]
    # truth pairs: (0,1)(0,2)(0,3)(1,2)(1,3)(2,3)  = 6
    # pred  pairs: (0,2)(0,1)(0,3)(2,1)(2,3)(1,3)
    # shared (same order): (0,1)(0,2)(0,3)(2,3)(1,3) = 5  -> 5/6
    val = pairs_f1([0, 2, 1, 3], [0, 1, 2, 3])
    assert math.isclose(val, 5.0 / 6.0, abs_tol=1e-9)
    assert val < 1.0


def test_pairs_f1_fully_reversed():
    # complete reversal: only ... no ordered pair survives -> 0
    assert pairs_f1([3, 2, 1, 0], [0, 1, 2, 3]) == 0.0


def test_pairs_f1_partial_overlap_same_length():
    # truth [0,1,2,3], pred [0,1,4,3] (POI 4 replaces 2)
    # truth pairs (6): (0,1)(0,2)(0,3)(1,2)(1,3)(2,3)
    # pred  pairs (6): (0,1)(0,4)(0,3)(1,4)(1,3)(4,3)
    # shared: (0,1)(0,3)(1,3) = 3 ; prec=rec=3/6=0.5 -> F1=0.5
    assert math.isclose(pairs_f1([0, 1, 4, 3], [0, 1, 2, 3]), 0.5, abs_tol=1e-9)


def test_pairs_f1_length_one_edge_case():
    assert pairs_f1([5], [5]) == 1.0      # identical singletons
    assert pairs_f1([5], [6]) == 0.0      # different singletons


def test_set_f1_order_agnostic():
    # same POIs, different order -> set-F1 is 1.0 even though pairs-F1 < 1
    assert set_f1([0, 2, 1, 3], [0, 1, 2, 3]) == 1.0
    assert pairs_f1([0, 2, 1, 3], [0, 1, 2, 3]) < 1.0


def test_set_f1_half_overlap():
    # pred {0,1,4,3} vs truth {0,1,2,3}: inter=3, prec=rec=3/4 -> F1=0.75
    assert math.isclose(set_f1([0, 1, 4, 3], [0, 1, 2, 3]), 0.75, abs_tol=1e-9)


def test_exact_match_and_feasibility():
    assert exact_match([0, 1, 2], [0, 1, 2]) == 1.0
    assert exact_match([0, 2, 1], [0, 1, 2]) == 0.0
    assert is_feasible([0, 1, 2], K=3) is True
    assert is_feasible([0, 1, 1], K=3) is False   # repeat
    assert is_feasible([0, 1, 2, 3], K=3) is False  # over budget


# --------------------------------------------------------------------------
# build_eval_queries
# --------------------------------------------------------------------------
def _toy_sessions_df():
    rows = []
    # session 1: user 0, length 4
    for i, p in enumerate([0, 1, 2, 3]):
        rows.append({"session_id": 1, "user_idx": 0, "poi_idx": p,
                     "timestamp": pd.Timestamp("2023-01-01 10:00") + pd.Timedelta(minutes=30 * i)})
    # session 2: user 1, length 2
    for i, p in enumerate([4, 5]):
        rows.append({"session_id": 2, "user_idx": 1, "poi_idx": p,
                     "timestamp": pd.Timestamp("2023-01-02 09:00") + pd.Timedelta(minutes=20 * i)})
    return pd.DataFrame(rows)


def test_build_eval_queries_shapes():
    qs = build_eval_queries(_toy_sessions_df(), fixed_end=True, min_len=2)
    assert len(qs) == 2
    q0 = next(q for q in qs if q.user_idx == 0)
    assert q0.start_poi == 0
    assert q0.end_poi == 3
    assert q0.K == 4
    assert q0.ground_truth == [0, 1, 2, 3]


def test_build_eval_queries_open_ended():
    qs = build_eval_queries(_toy_sessions_df(), fixed_end=False, min_len=3)
    assert len(qs) == 1            # only the length-4 session survives min_len=3
    assert qs[0].end_poi is None


# --------------------------------------------------------------------------
# decode invariants — tiny random model
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_setup():
    torch.manual_seed(0)
    n_pois, n_users = 40, 5
    model = NextPOIModel(n_pois, n_users).to(DEVICE).eval()
    # connected-ish small graph
    src = list(range(n_pois)) + list(range(0, n_pois - 1))
    dst = list(range(n_pois)) + list(range(1, n_pois))
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long, device=DEVICE)
    rng = np.random.default_rng(0)
    poi_coords = rng.uniform(low=[40.0, -74.1], high=[41.0, -73.7], size=(n_pois, 2))
    return model, edge_index, poi_coords


def _query(start, end, K, gt=None):
    return ItineraryQuery(user_idx=0, start_poi=start, end_poi=end, K=K,
                          ground_truth=gt or list(range(K)))


def test_greedy_loop_free_and_length(tiny_setup):
    model, edge_index, coords = tiny_setup
    q = _query(start=3, end=7, K=6)
    route = rollout_greedy(model, q, edge_index, coords, DEVICE)
    assert len(route) == q.K
    assert len(set(route)) == len(route)        # no repeats
    assert route[0] == 3                          # starts at start


def test_greedy_fixed_end_is_last(tiny_setup):
    model, edge_index, coords = tiny_setup
    q = _query(start=3, end=7, K=6)
    route = rollout_greedy(model, q, edge_index, coords, DEVICE)
    assert route[-1] == 7                          # end POI last


def test_fixed_end_reserved_not_consumed_early(tiny_setup):
    # The end POI must appear exactly once, at the last position — never in the
    # middle — even across many start/end/K combinations (regression: the end
    # used to be pickable early by greedy/beam).
    model, edge_index, coords = tiny_setup
    for start, end, K in [(3, 7, 6), (0, 1, 5), (10, 2, 7), (5, 6, 4), (8, 9, 3)]:
        q = _query(start, end, K)
        for route in (
            rollout_greedy(model, q, edge_index, coords, DEVICE),
            rollout_beam(model, q, edge_index, coords, DEVICE, beam=4),
        ):
            assert route[-1] == end
            assert route.count(end) == 1
            assert end not in route[:-1]


def test_greedy_equals_beam1(tiny_setup):
    model, edge_index, coords = tiny_setup
    q = _query(start=2, end=9, K=5)
    g = rollout_greedy(model, q, edge_index, coords, DEVICE)
    b = rollout_beam(model, q, edge_index, coords, DEVICE, beam=1)
    assert g == b


def test_beam_loop_free_and_length(tiny_setup):
    model, edge_index, coords = tiny_setup
    q = _query(start=1, end=8, K=6)
    route = rollout_beam(model, q, edge_index, coords, DEVICE, beam=4)
    assert len(route) == q.K
    assert len(set(route)) == len(route)
    assert route[0] == 1 and route[-1] == 8


def test_degenerate_k1_returns_start(tiny_setup):
    model, edge_index, coords = tiny_setup
    q = _query(start=5, end=5, K=1, gt=[5])
    assert rollout_greedy(model, q, edge_index, coords, DEVICE) == [5]
    assert rollout_beam(model, q, edge_index, coords, DEVICE, beam=3) == [5]


def test_length2_query_is_trivially_perfect(tiny_setup):
    # K=2 with fixed end -> route forced to [start, end] == ground truth -> pairs-F1 = 1.0
    model, edge_index, coords = tiny_setup
    q = _query(start=4, end=9, K=2, gt=[4, 9])
    route = rollout_greedy(model, q, edge_index, coords, DEVICE)
    assert route == [4, 9]
    assert pairs_f1(route, q.ground_truth) == 1.0


def test_evaluate_itinerary_runs_and_keys(tiny_setup):
    model, edge_index, coords = tiny_setup
    queries = [_query(3, 7, 6), _query(2, 9, 5, gt=[2, 0, 1, 8, 9]), _query(4, 9, 2, gt=[4, 9])]
    m = evaluate_itinerary(model, queries, edge_index, coords, DEVICE, decoder="greedy")
    for k in ("pairs-F1", "set-F1", "exact-match", "feasibility", "n"):
        assert k in m
    assert m["n"] == 3.0
    assert m["feasibility"] == 1.0                # loop-free by construction
    # length>=3 subset drops the K=2 query
    m3 = evaluate_itinerary(model, queries, edge_index, coords, DEVICE,
                            decoder="greedy", min_len=3)
    assert m3["n"] == 2.0
