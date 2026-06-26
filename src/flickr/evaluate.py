"""Leave-one-trajectory-out evaluation harness for the Flickr datasets.

Implements the *canonical* trip-recommendation protocol so our numbers are
directly comparable to the literature (Chen 2016 and successors):

1. Evaluate only trajectories of length >= ``min_len`` (default 3) — a length-2
   query is the trivial ``[start, end]`` with no order to recover.
2. The query gives the **first and last POI** (origin + destination) and the
   desired length K; the recommender must produce the ordered POIs in between.
3. **Leave-one-trajectory-out cross-validation**: each length>=3 trajectory is
   held out once for testing while all *other* trajectories train the model.
4. Metrics: point-F1 (``set_f1``) and order-aware **pairs-F1** — reusing the
   existing, unit-tested implementations in :mod:`src.itinerary.eval_itinerary`
   (verified to match Chen 2016 by ``tests/test_flickr.py``).

The recommender is supplied as a *factory* ``factory(city, train_trajs)`` so a
fresh model is fit on each fold's training set — the held-out trajectory never
leaks into training. All functions are pure; configuration is explicit.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Optional, Protocol, Tuple

from src.flickr.data import FlickrCity, Trajectory, trajectories_min_len
from src.itinerary.eval_itinerary import exact_match, pairs_f1, set_f1
from src.itinerary.query import ItineraryQuery


class Recommender(Protocol):
    """A fitted recommender: maps a query to an ordered route of POI indices."""

    def recommend(self, query: ItineraryQuery) -> List[int]:
        """Return an ordered list of POI indices of length ``query.K``.

        The route must start at ``query.start_poi`` and (if ``query.end_poi`` is
        not None) end at ``query.end_poi``, and be loop-free.
        """
        ...


#: A factory that fits a recommender on one fold's training trajectories.
RecommenderFactory = Callable[[FlickrCity, List[Trajectory]], Recommender]


def make_query(traj: Trajectory, fixed_end: bool = True) -> ItineraryQuery:
    """Build the itinerary query for a ground-truth trajectory.

    Args:
        traj: the held-out trajectory (its POIs are the ground truth).
        fixed_end: if True the query fixes the last POI (origin+destination
            given, the standard protocol); if False the end is open.

    Returns:
        An :class:`ItineraryQuery` with ``K = len(traj)`` and
        ``ground_truth = list(traj.pois)``.
    """
    pois = list(traj.pois)
    return ItineraryQuery(
        user_idx=traj.user_idx,
        start_poi=pois[0],
        end_poi=pois[-1] if fixed_end else None,
        K=len(pois),
        ground_truth=pois,
    )


def eligible_eval_trajectories(
    city: FlickrCity, min_len: int = 3
) -> Tuple[List[Trajectory], int]:
    """Length>=min_len trajectories that satisfy the pairs-F1 assumptions.

    The canonical protocol (and Chen 2016's ``calc_pairsF1``) assumes a
    **loop-free** ground truth with **distinct** endpoints — a route that
    revisits a POI, or returns to its origin (``start == end``), makes the
    ordered-pair set self-contradictory and the fixed-end query ill-defined.
    The Flickr ``traj-*.csv`` data is entirely loop-free, so this drops nothing
    in practice, but it makes the assumption explicit and guards the metric.

    Args:
        city: the loaded city.
        min_len: minimum trajectory length (default 3).

    Returns:
        ``(eligible, n_dropped)`` — the eligible trajectories and how many
        length>=min_len trajectories were excluded as non-loop-free / round-trip.
    """
    eligible: List[Trajectory] = []
    dropped = 0
    for t in trajectories_min_len(city, min_len):
        p = t.pois
        if len(set(p)) == len(p) and p[0] != p[-1]:
            eligible.append(t)
        else:
            dropped += 1
    return eligible, dropped


def loocv_splits(
    test_trajs: List[Trajectory], pool: List[Trajectory]
) -> Iterator[Tuple[Trajectory, List[Trajectory]]]:
    """Yield ``(held_out, training_trajectories)`` for each test trajectory.

    The held-out trajectory is removed from ``pool`` by object identity, so the
    training set for each fold excludes exactly the trajectory under test (and
    nothing else).

    Args:
        test_trajs: trajectories to be held out one at a time (length>=min_len).
        pool: the full pool of trajectories to draw training data from (either
            all trajectories, or just the length>=min_len ones).

    Yields:
        ``(test, train)`` where ``test`` is one held-out trajectory and ``train``
        is ``pool`` minus ``test``.
    """
    for test in test_trajs:
        train = [t for t in pool if t is not test]
        yield test, train


def evaluate_loocv(
    city: FlickrCity,
    factory: RecommenderFactory,
    *,
    min_len: int = 3,
    fixed_end: bool = True,
    train_pool: str = "ge3",
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, float]:
    """Run leave-one-trajectory-out CV and average F1 + pairs-F1 for one city.

    Args:
        city: the loaded :class:`FlickrCity`.
        factory: ``factory(city, train_trajs)`` -> fitted :class:`Recommender`.
        min_len: minimum trajectory length to evaluate (default 3, the protocol).
        fixed_end: give the destination POI in the query (default True).
        train_pool: ``"ge3"`` (default, the faithful protocol) trains each fold
            on the *other length>=min_len* trajectories — the same population the
            eval set is drawn from, matching Chen 2016's released folds. ``"all"``
            trains on every other trajectory (length 1, 2, 3, ...); it is a
            documented deviation that lets length<3 visits/transitions leak extra
            popularity/co-visit signal the papers' folds never see. Both exclude
            the held-out trajectory.
        on_progress: optional callback ``(done, total)`` invoked each fold.

    Returns:
        Dict with ``F1``, ``pairs-F1``, ``exact-match`` (means over folds),
        ``n`` (number of evaluated trajectories), ``city`` (name), the config
        echoes ``min_len`` / ``train_pool``, and ``n_dropped_non_loopfree`` (how
        many length>=min_len trajectories were excluded as non-loop-free /
        round-trip; 0 on the standard Flickr data).
    """
    test_trajs, n_dropped = eligible_eval_trajectories(city, min_len)
    pool = city.trajectories if train_pool == "all" else test_trajs

    sums = {"F1": 0.0, "pairs-F1": 0.0, "exact-match": 0.0}
    n = 0
    total = len(test_trajs)
    for test, train in loocv_splits(test_trajs, pool):
        rec = factory(city, train)
        q = make_query(test, fixed_end=fixed_end)
        route = rec.recommend(q)
        sums["F1"] += set_f1(route, q.ground_truth)
        sums["pairs-F1"] += pairs_f1(route, q.ground_truth)
        sums["exact-match"] += exact_match(route, q.ground_truth)
        n += 1
        if on_progress is not None and (n % 50 == 0 or n == total):
            on_progress(n, total)

    out: Dict[str, float] = {k: (v / n if n else 0.0) for k, v in sums.items()}
    out["n"] = float(n)
    out["city"] = city.name  # type: ignore[assignment]
    out["min_len"] = float(min_len)
    out["train_pool"] = train_pool  # type: ignore[assignment]
    out["n_dropped_non_loopfree"] = float(n_dropped)
    return out
