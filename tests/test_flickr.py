"""Tests for the Flickr (Strategy D) layer: loader, trajectory construction,
pairs-F1 protocol faithfulness, leave-one-out splitter, and baseline validity.

The pairs-F1 cases pin the metric against an independent re-implementation of
Chen 2016's reference ``calc_pairsF1`` (the published numbers are computed with
that definition; a subtly different one invalidates every comparison). Loader
and splitter tests use small synthetic CSVs written to a temp dir — no real data
download is needed, so the suite runs offline on CPU.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest
import torch

from src.flickr.baselines import (
    MarkovRecommender,
    PopularityRecommender,
    RandomRecommender,
    classical_factories,
    fit_log_transition,
    markov_factory,
    popularity_factory,
    random_factory,
)
from src.flickr.data import (
    Trajectory,
    _collapse_consecutive,
    load_city,
    trajectories_min_len,
)
from src.flickr.evaluate import evaluate_loocv, loocv_splits, make_query
from src.flickr.published import PUBLISHED, published_for_city
from src.itinerary.eval_itinerary import pairs_f1, set_f1
from src.itinerary.query import ItineraryQuery


# ===========================================================================
# Synthetic CSV fixtures (both supported on-disk formats)
# ===========================================================================
_POI_CHEN = """poiID,poiCat,poiLon,poiLat
1,Sport,-79.3,43.60
2,Sport,-79.4,43.70
3,Cultural,-79.5,43.65
4,Cultural,-79.2,43.62
5,Park,-79.1,43.71
"""

# trajID 1 user A: 1,2,3 ; trajID 2 user B: 3,4,5,1 ; trajID 3 user A: 1,1,2,3
# (consecutive dup collapses to 1,2,3) ; trajID 4 user C: 5,4 (length 2) ;
# trajID 5 user B: 1,3,5,2,4. Rows are deliberately shuffled in time to test sorting.
_TRAJ_CHEN = """userID,trajID,poiID,startTime,endTime,#photo,trajLen,poiDuration
A,1,2,200,200,1,3,0
A,1,1,100,100,1,3,0
A,1,3,300,300,1,3,0
B,2,3,100,100,1,4,0
B,2,4,200,200,1,4,0
B,2,5,300,300,1,4,0
B,2,1,400,400,1,4,0
A,3,1,100,100,1,3,0
A,3,1,150,150,1,3,0
A,3,2,200,200,1,3,0
A,3,3,300,300,1,3,0
C,4,5,100,100,1,2,0
C,4,4,200,200,1,2,0
B,5,1,100,100,1,5,0
B,5,3,200,200,1,5,0
B,5,5,300,300,1,5,0
B,5,2,400,400,1,5,0
B,5,4,500,500,1,5,0
"""

_POI_LIM = """poiID,poiName,lat,lon,theme
1,Alpha,43.60,-79.3,Sport
2,Beta,43.70,-79.4,Sport
3,Gamma,43.65,-79.5,Cultural
4,Delta,43.62,-79.2,Cultural
5,Eps,43.71,-79.1,Park
"""

_VISITS_LIM = """photoID,userID,dateTaken,poiID,poiTheme,poiFreq,seqID
p1,A,100,1,Sport,5,1
p2,A,200,2,Sport,5,1
p3,A,300,3,Cultural,5,1
p4,B,100,3,Cultural,5,2
p5,B,200,4,Cultural,5,2
p6,B,300,5,Park,5,2
"""


@pytest.fixture()
def chen_city_dir(tmp_path):
    (tmp_path / "poi-Test.csv").write_text(_POI_CHEN, encoding="utf-8")
    (tmp_path / "traj-Test.csv").write_text(_TRAJ_CHEN, encoding="utf-8")
    return str(tmp_path)


@pytest.fixture()
def lim_city_dir(tmp_path):
    (tmp_path / "POI-Test.csv").write_text(_POI_LIM, encoding="utf-8")
    (tmp_path / "userVisits-Test.csv").write_text(_VISITS_LIM, encoding="utf-8")
    return str(tmp_path)


# ===========================================================================
# Loader + trajectory construction
# ===========================================================================
def test_collapse_consecutive_duplicates():
    assert _collapse_consecutive([1, 1, 2, 2, 2, 3]) == [1, 2, 3]
    assert _collapse_consecutive([1, 2, 1]) == [1, 2, 1]      # non-consecutive kept
    assert _collapse_consecutive([5]) == [5]


def test_load_city_chen_format(chen_city_dir):
    c = load_city(chen_city_dir, "Test")
    assert c.n_pois == 5
    assert c.poi_coords.shape == (5, 2)
    # poi ids 1..5 map to contiguous 0..4; coords are [lat, lon]
    assert c.poi2idx == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    assert math.isclose(c.poi_coords[0, 0], 43.60) and math.isclose(c.poi_coords[0, 1], -79.3)
    # 5 trajectories; users A,B,C -> 3 users
    assert len(c.trajectories) == 5
    assert c.n_users == 3
    by_id = {t.traj_id: t for t in c.trajectories}
    # ordered by startTime, mapped to indices
    assert by_id[1].pois == (0, 1, 2)              # POIs (time-sorted) 1,2,3 -> idx 0,1,2
    assert by_id[2].pois == (2, 3, 4, 0)           # 3,4,5,1 -> idx 2,3,4,0
    # consecutive-dup trajectory 1,1,2,3 collapses to 1,2,3
    assert by_id[3].pois == (0, 1, 2)
    assert by_id[4].pois == (4, 3)                 # length-2 trajectory survives load


def test_trajectories_min_len_filters(chen_city_dir):
    c = load_city(chen_city_dir, "Test")
    ge3 = trajectories_min_len(c, 3)
    # trajectories 1,2,3,5 have length >=3; trajectory 4 (length 2) dropped
    assert len(ge3) == 4
    assert all(len(t.pois) >= 3 for t in ge3)


def test_load_city_lim_format(lim_city_dir):
    c = load_city(lim_city_dir, "Test")
    assert c.n_pois == 5
    assert len(c.trajectories) == 2
    by_id = {t.traj_id: t for t in c.trajectories}
    assert by_id[1].pois == (0, 1, 2)              # seqID 1: POIs 1,2,3
    assert by_id[2].pois == (2, 3, 4)              # seqID 2: POIs 3,4,5
    # lat/lon parsed from the lat,lon columns
    assert math.isclose(c.poi_coords[0, 0], 43.60) and math.isclose(c.poi_coords[0, 1], -79.3)


def test_load_city_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_city(str(tmp_path), "Nowhere")


# ===========================================================================
# pairs-F1 protocol — pinned against Chen 2016's reference calc_pairsF1
# ===========================================================================
def _chen_calc_pairs_f1(y: List[int], y_hat: List[int]) -> float:
    """Independent re-implementation of Chen 2016's reference ``calc_pairsF1``.

    ``y`` is the (loop-free) ground truth; ``y_hat`` the prediction. Counts
    ordered pairs of ``y_hat`` that respect the visiting order in ``y``.
    """
    n, nr = len(y), len(y_hat)
    n0 = n * (n - 1) / 2.0
    n0r = nr * (nr - 1) / 2.0
    order = {p: i for i, p in enumerate(y)}
    nc = 0
    for i in range(nr):
        for j in range(i + 1, nr):
            a, b = y_hat[i], y_hat[j]
            if a in order and b in order and a != b and order[a] < order[b]:
                nc += 1
    if n0 == 0 or n0r == 0:
        return 1.0 if list(y) == list(y_hat) else 0.0
    prec, rec = nc / n0r, nc / n0
    return 0.0 if nc == 0 else 2.0 * prec * rec / (prec + rec)


def test_pairs_f1_hand_checked():
    assert pairs_f1([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert math.isclose(pairs_f1([0, 2, 1, 3], [0, 1, 2, 3]), 5.0 / 6.0)
    assert pairs_f1([3, 2, 1, 0], [0, 1, 2, 3]) == 0.0


def test_pairs_f1_matches_chen_reference_random_loopfree():
    rng = np.random.default_rng(0)
    pool = list(range(12))
    for _ in range(400):
        k = int(rng.integers(3, 8))
        truth = list(rng.permutation(pool))[:k]               # loop-free
        pred = list(rng.permutation(pool))[:k]                # same-length, loop-free
        # our pairs_f1(pred, truth) must equal chen calc_pairsF1(y=truth, y_hat=pred)
        assert math.isclose(pairs_f1(pred, truth), _chen_calc_pairs_f1(truth, pred), abs_tol=1e-12)


def test_pairs_f1_matches_chen_reference_endpoints_given():
    # The protocol fixes start+end: build pred that shares endpoints with truth.
    rng = np.random.default_rng(1)
    for _ in range(300):
        k = int(rng.integers(3, 7))
        mids = list(rng.permutation(list(range(1, 11))))
        truth = [0] + mids[: k - 2] + [99]
        pred_mids = list(rng.permutation(list(range(1, 11))))
        pred = [0] + pred_mids[: k - 2] + [99]
        assert math.isclose(pairs_f1(pred, truth), _chen_calc_pairs_f1(truth, pred), abs_tol=1e-12)


def test_set_f1_matches_chen_reference():
    # Chen's point-F1 for loop-free equal-length sequences == our set_f1.
    rng = np.random.default_rng(2)
    for _ in range(200):
        k = int(rng.integers(3, 7))
        truth = list(rng.permutation(list(range(10))))[:k]
        pred = list(rng.permutation(list(range(10))))[:k]
        inter = len(set(pred) & set(truth))
        prec, rec = inter / len(pred), inter / len(truth)
        chen_f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        assert math.isclose(set_f1(pred, truth), chen_f1, abs_tol=1e-12)


# ===========================================================================
# Leave-one-out splitter
# ===========================================================================
def test_loocv_splits_holds_each_out_once():
    trajs = [
        Trajectory(traj_id=i, user_idx=0, pois=tuple(range(3)))
        for i in range(5)
    ]
    splits = list(loocv_splits(trajs, trajs))
    assert len(splits) == 5
    for held, train in splits:
        assert held not in train                 # identity exclusion
        assert len(train) == 4
        assert all(t is not held for t in train)
    # every trajectory is the held-out one exactly once
    assert {id(h) for h, _ in splits} == {id(t) for t in trajs}


def test_loocv_train_pool_all_vs_ge3(chen_city_dir):
    # With train_pool="all", the length-2 trajectory is available for training;
    # with "ge3" it is not. Both still hold out only length>=3 test trajectories.
    c = load_city(chen_city_dir, "Test")
    seen_train_sizes = {"all": set(), "ge3": set()}
    for pool in ("all", "ge3"):
        test_trajs = trajectories_min_len(c, 3)
        base = c.trajectories if pool == "all" else test_trajs
        for _held, train in loocv_splits(test_trajs, base):
            seen_train_sizes[pool].add(len(train))
    # "all" folds train on 4 others (5 total -1); "ge3" on 3 others (4 ge3 -1)
    assert seen_train_sizes["all"] == {4}
    assert seen_train_sizes["ge3"] == {3}


# ===========================================================================
# Baseline validity (loop-free, length K, correct endpoints, determinism)
# ===========================================================================
def _valid_route(route: List[int], q: ItineraryQuery) -> bool:
    return (
        len(route) == q.K
        and len(set(route)) == len(route)        # loop-free
        and route[0] == q.start_poi
        and (q.end_poi is None or route[-1] == q.end_poi)
    )


def _toy_train() -> List[Trajectory]:
    return [
        Trajectory(0, 0, (0, 1, 2, 3)),
        Trajectory(1, 1, (0, 2, 3, 4)),
        Trajectory(2, 0, (1, 2, 4)),
        Trajectory(3, 1, (0, 1, 4, 3)),
    ]


def test_random_recommender_valid_and_deterministic():
    rec = RandomRecommender(n_pois=8, seed=0)
    q = ItineraryQuery(user_idx=0, start_poi=0, end_poi=7, K=5, ground_truth=[0, 1, 2, 3, 7])
    r1 = rec.recommend(q)
    r2 = rec.recommend(q)
    assert _valid_route(r1, q)
    assert r1 == r2                               # deterministic for a given query


def test_popularity_recommender_orders_by_popularity():
    pop = np.array([0.0, 9.0, 1.0, 5.0, 2.0])     # POI 1 most popular, then 3, then 4, 2
    rec = PopularityRecommender(pop)
    q = ItineraryQuery(user_idx=0, start_poi=0, end_poi=4, K=4, ground_truth=[0, 1, 3, 4])
    r = rec.recommend(q)
    assert _valid_route(r, q)
    # K=4, start 0, end 4 -> 2 middle slots filled by the two most popular of {1,2,3}
    assert r == [0, 1, 3, 4]                       # pop order 1 (9) then 3 (5)


def test_markov_greedy_and_beam_valid():
    n = 5
    logtrans = fit_log_transition(n, _toy_train(), alpha=0.1)
    q = ItineraryQuery(user_idx=0, start_poi=0, end_poi=3, K=4, ground_truth=[0, 1, 2, 3])
    for mode in ("greedy", "beam"):
        rec = MarkovRecommender(logtrans, mode=mode, beam=4)
        r = rec.recommend(q)
        assert _valid_route(r, q)


def test_markov_greedy_stays_loop_free_when_K_exceeds_vocab():
    # Regression: when no unvisited non-end POI remains, the greedy decoder must
    # NOT append a phantom (already-visited) POI — it must stop, staying loop-free.
    n = 3
    logtrans = fit_log_transition(
        n, [Trajectory(0, 0, (0, 1, 2)), Trajectory(1, 0, (2, 1, 0))], alpha=0.1
    )
    q = ItineraryQuery(user_idx=0, start_poi=0, end_poi=2, K=4, ground_truth=[0, 1, 2])
    r = MarkovRecommender(logtrans, mode="greedy").recommend(q)
    assert len(set(r)) == len(r)            # loop-free (the fixed defect)
    assert r[0] == 0
    assert len(r) <= q.K


def test_eligible_eval_trajectories_drops_loops_and_roundtrips():
    from src.flickr.data import FlickrCity
    from src.flickr.evaluate import eligible_eval_trajectories

    trajs = [
        Trajectory(0, 0, (0, 1, 2)),        # ok
        Trajectory(1, 0, (0, 1, 0)),        # round-trip start==end -> dropped
        Trajectory(2, 0, (0, 1, 2, 1)),     # non-loop-free (1 repeats) -> dropped
        Trajectory(3, 0, (3, 4, 5)),        # ok
        Trajectory(4, 0, (0, 1)),           # length 2 -> not in >=3 set at all
    ]
    city = FlickrCity(
        name="T", n_pois=6, n_users=1, poi_coords=np.zeros((6, 2)),
        poi_cat=[""] * 6, trajectories=trajs, poi2idx={}, user2idx={},
    )
    elig, dropped = eligible_eval_trajectories(city, 3)
    assert dropped == 2
    assert {t.traj_id for t in elig} == {0, 3}


def test_evaluate_loocv_reports_zero_dropped_on_clean_city(chen_city_dir):
    c = load_city(chen_city_dir, "Test")
    m = evaluate_loocv(c, random_factory(seed=0), min_len=3, train_pool="ge3")
    assert m["n_dropped_non_loopfree"] == 0.0   # Flickr/toy data is loop-free


def test_fit_log_transition_shapes_and_normalisation():
    n = 5
    lt = fit_log_transition(n, _toy_train(), alpha=0.1)
    assert lt.shape == (n, n)
    probs = np.exp(lt)
    assert np.allclose(probs.sum(axis=1), 1.0)    # each row is a distribution


def test_classical_factories_run_on_toy_city(chen_city_dir):
    c = load_city(chen_city_dir, "Test")
    for name, fac in classical_factories().items():
        m = evaluate_loocv(c, fac, min_len=3, train_pool="all")
        assert m["n"] == 4.0                       # 4 length>=3 trajectories
        assert 0.0 <= m["pairs-F1"] <= 1.0
        assert 0.0 <= m["F1"] <= 1.0


# ===========================================================================
# Neural pointer — forward shapes + decode invariants (no training, CPU)
# ===========================================================================
def _tiny_pointer_setup():
    from src.flickr.data import FlickrCity
    from src.flickr.pointer import FlickrPointerNet, build_poi_graph

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n = 9
    coords = rng.uniform(low=[43.6, -79.5], high=[43.75, -79.1], size=(n, 2))
    city = FlickrCity(
        name="Tiny", n_pois=n, n_users=2, poi_coords=coords,
        poi_cat=[""] * n, trajectories=[], poi2idx={}, user2idx={},
    )
    train = _toy_train()
    edge_index = build_poi_graph(city, train, knn_k=3, covisit_threshold=1)
    model = FlickrPointerNet(n, dim=16).eval()
    return city, model, edge_index, torch.device("cpu")


def test_build_poi_graph_symmetric_with_self_loops():
    city, _model, edge_index, _dev = _tiny_pointer_setup()
    assert edge_index.shape[0] == 2
    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    # self-loop on every node
    for i in range(city.n_pois):
        assert (i, i) in edges
    # symmetric: (a,b) present => (b,a) present
    for a, b in edges:
        assert (b, a) in edges


def test_pointer_forward_shapes():
    _city, model, edge_index, _dev = _tiny_pointer_setup()
    poi_seq = torch.tensor([[0, 1, 2, 3], [0, 2, 4, 0]])   # 2nd is length 3 (pad 0)
    lengths = torch.tensor([4, 3])
    logits, targets, mask = model(poi_seq, lengths, edge_index)
    assert logits.shape == (2, 3, model.n_pois)
    assert targets.shape == (2, 3)
    assert mask.tolist() == [[True, True, True], [True, True, False]]


def test_pointer_decode_invariants():
    from src.flickr.pointer import pointer_decode

    _city, model, edge_index, dev = _tiny_pointer_setup()
    for start, end, K in [(0, 7, 5), (1, 8, 4), (2, 3, 6)]:
        q = ItineraryQuery(user_idx=0, start_poi=start, end_poi=end, K=K,
                           ground_truth=list(range(K)))
        for beam in (1, 3):
            r = pointer_decode(model, q, edge_index, dev, beam=beam)
            assert _valid_route(r, q)


# ===========================================================================
# Published-results module sanity
# ===========================================================================
def test_published_module_is_sane():
    assert len(PUBLISHED) > 40
    edin = published_for_city("Edinburgh", "pairs_F1")
    assert "Random" in edin and "SelfTrip" in edin
    # every comparable pairs-F1 in [0,1]
    for p in PUBLISHED:
        if p.pairs_F1 is not None:
            assert 0.0 <= p.pairs_F1 <= 1.0
        if p.F1 is not None:
            assert 0.0 <= p.F1 <= 1.0
