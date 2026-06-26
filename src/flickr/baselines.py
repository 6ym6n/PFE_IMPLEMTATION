"""Classical trip-recommendation baselines for the Flickr datasets.

Faithful, light re-implementations of the standard baselines whose numbers the
literature reports (Chen 2016):

* :class:`RandomRecommender`    — random order of intermediate POIs  (Chen "Random")
* :class:`PopularityRecommender`— most-visited POIs, popularity order (Chen "PoiPopularity")
* :class:`MarkovRecommender`    — first-order POI transition model, decoded either
    greedily (≈ Chen "Markov") or by beam search over transition log-prob
    (≈ Chen "MarkovPath", the loop-free maximum-likelihood path).

All operate on the endpoint-constrained, length-budget query
(:class:`~src.itinerary.query.ItineraryQuery`): produce an ordered, loop-free
route of length ``K`` from ``start_poi`` to ``end_poi``. They reuse the same
reserve-the-end / no-revisit discipline as the Foursquare itinerary decoders.

PoiRank and Rank+Markov are *not* re-implemented here — they require a per-query
rankSVM whose faithful reproduction is out of scope; their published numbers are
cited directly (see :mod:`src.flickr.published`). The four baselines above map
cleanly onto Chen's Random / PoiPopularity / Markov / MarkovPath and serve as the
reproducibility check for the harness.

No global state; every model is fit from explicit training trajectories.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from src.flickr.data import FlickrCity, Trajectory
from src.itinerary.query import ItineraryQuery


# ---------------------------------------------------------------------------
# Route assembly helper (shared discipline: start first, fixed end last, no loop)
# ---------------------------------------------------------------------------
def _assemble(start: int, end: Optional[int], K: int, middle: Sequence[int]) -> List[int]:
    """Glue an ordered list of intermediate POIs into a full route.

    Args:
        start: first POI (always position 0).
        end: last POI, or None for an open-ended route.
        K: target route length.
        middle: ordered intermediate POIs (must exclude start and end and be
            loop-free); only the first ``K-2`` (or ``K-1`` if open) are used.

    Returns:
        Route of length ``K`` (or shorter only if ``middle`` is too short),
        starting at ``start`` and, when fixed, ending at ``end``.
    """
    if end is None:
        return [start] + list(middle[: K - 1])
    return [start] + list(middle[: K - 2]) + [end]


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------
class RandomRecommender:
    """Order the intermediate POIs uniformly at random (Chen 2016 "Random").

    Deterministic given the query: the RNG is seeded from (start, end, K) so a
    leave-one-out run is reproducible.

    Args:
        n_pois: vocabulary size |V|.
        seed: base seed mixed with the query to seed the per-query RNG.
    """

    def __init__(self, n_pois: int, seed: int = 0) -> None:
        self.n_pois = n_pois
        self.seed = seed

    def recommend(self, query: ItineraryQuery) -> List[int]:
        """Random loop-free route of length ``K`` from start to end."""
        end = query.end_poi
        blocked = {query.start_poi}
        if end is not None:
            blocked.add(end)
        candidates = [p for p in range(self.n_pois) if p not in blocked]
        need = query.K - (2 if end is not None else 1)
        need = max(0, min(need, len(candidates)))
        mix = (self.seed * 2654435761 + query.start_poi * 1000003
               + (end if end is not None else -1) * 9176 + query.K) & 0xFFFFFFFF
        rng = np.random.default_rng(mix)
        middle = rng.choice(candidates, size=need, replace=False).tolist() if need else []
        return _assemble(query.start_poi, end, query.K, [int(x) for x in middle])


def random_factory(seed: int = 0):
    """Return a :data:`~src.flickr.evaluate.RecommenderFactory` for Random."""

    def factory(city: FlickrCity, train: List[Trajectory]) -> RandomRecommender:
        return RandomRecommender(city.n_pois, seed=seed)

    return factory


# ---------------------------------------------------------------------------
# Popularity (PoiPopularity)
# ---------------------------------------------------------------------------
class PopularityRecommender:
    """Pick + order the most-visited POIs (Chen 2016 "PoiPopularity").

    Popularity = number of visits to a POI across the training trajectories.
    The intermediate POIs are the top-(K-2) most popular (excluding start/end),
    ordered by descending popularity (ties broken by POI index for determinism).

    Args:
        popularity: length-|V| array of training visit counts per POI index.
    """

    def __init__(self, popularity: np.ndarray) -> None:
        self.popularity = popularity

    def recommend(self, query: ItineraryQuery) -> List[int]:
        """Popularity-ordered loop-free route of length ``K``."""
        end = query.end_poi
        blocked = {query.start_poi}
        if end is not None:
            blocked.add(end)
        cand = [p for p in range(len(self.popularity)) if p not in blocked]
        # sort by popularity desc, then index asc (stable, deterministic)
        cand.sort(key=lambda p: (-float(self.popularity[p]), p))
        need = query.K - (2 if end is not None else 1)
        middle = cand[: max(0, need)]
        return _assemble(query.start_poi, end, query.K, middle)


def _visit_counts(n_pois: int, trajs: Sequence[Trajectory]) -> np.ndarray:
    """Per-POI visit counts over a set of trajectories.

    Args:
        n_pois: vocabulary size.
        trajs: trajectories to count over.

    Returns:
        (n_pois,) float array of visit counts.
    """
    counts = np.zeros(n_pois, dtype=float)
    for t in trajs:
        for p in t.pois:
            counts[p] += 1.0
    return counts


def popularity_factory():
    """Return a factory for the PoiPopularity baseline."""

    def factory(city: FlickrCity, train: List[Trajectory]) -> PopularityRecommender:
        return PopularityRecommender(_visit_counts(city.n_pois, train))

    return factory


# ---------------------------------------------------------------------------
# Markov transition model (greedy ≈ "Markov", beam ≈ "MarkovPath")
# ---------------------------------------------------------------------------
class MarkovRecommender:
    """First-order POI transition model, decoded greedily or by beam search.

    The transition log-probabilities ``log P(j | i)`` are estimated from the
    consecutive POI pairs in the training trajectories with additive (Laplace)
    smoothing, so unseen transitions still get a small, non-zero score.

    * ``mode="greedy"`` reproduces Chen's **Markov**: from the current POI, jump
      to the highest-probability unvisited POI, reserving the fixed end for last.
    * ``mode="beam"`` approximates Chen's **MarkovPath**: beam search for the
      loop-free path of length ``K`` from start to end with the greatest summed
      transition log-probability.

    Args:
        logtrans: (|V|, |V|) array of ``log P(j | i)``.
        mode: ``"greedy"`` or ``"beam"``.
        beam: beam width when ``mode == "beam"``.
    """

    def __init__(self, logtrans: np.ndarray, mode: str = "greedy", beam: int = 5) -> None:
        self.logtrans = logtrans
        self.mode = mode
        self.beam = beam

    def recommend(self, query: ItineraryQuery) -> List[int]:
        """Decode a loop-free route of length ``K`` from start to end."""
        if self.mode == "beam":
            return self._beam(query)
        return self._greedy(query)

    def _greedy(self, query: ItineraryQuery) -> List[int]:
        start, end, K = query.start_poi, query.end_poi, query.K
        route = [start]
        visited = {start}
        for step in range(1, K):
            is_last = step == K - 1
            if is_last and end is not None and end not in visited:
                nxt = end
            else:
                row = self.logtrans[route[-1]].copy()
                row[list(visited)] = -np.inf
                if end is not None and not is_last:
                    row[end] = -np.inf
                if not np.isfinite(row).any():
                    break  # no unvisited POI left (K > available); stay loop-free
                nxt = int(np.argmax(row))
            route.append(nxt)
            visited.add(nxt)
        return route

    def _beam(self, query: ItineraryQuery) -> List[int]:
        start, end, K = query.start_poi, query.end_poi, query.K
        if K <= 1:
            return [start]
        # beam entries: (route, score, frozenset visited)
        beams = [([start], 0.0, frozenset({start}))]
        for step in range(1, K):
            is_last = step == K - 1
            cand = []
            for route, score, visited in beams:
                row = self.logtrans[route[-1]]
                if is_last and end is not None and end not in visited:
                    cand.append((route + [end], score + float(row[end]),
                                 visited | {end}))
                    continue
                # candidate next POIs: unvisited, end reserved until last
                order = np.argsort(row)[::-1]
                taken = 0
                for j in order:
                    j = int(j)
                    if j in visited:
                        continue
                    if end is not None and not is_last and j == end:
                        continue
                    cand.append((route + [j], score + float(row[j]), visited | {j}))
                    taken += 1
                    if taken >= self.beam:
                        break
            if not cand:
                break
            cand.sort(key=lambda t: t[1], reverse=True)
            beams = cand[: self.beam]
        return beams[0][0]


def fit_log_transition(n_pois: int, trajs: Sequence[Trajectory], alpha: float = 0.1) -> np.ndarray:
    """Estimate smoothed ``log P(j | i)`` from training trajectories.

    Args:
        n_pois: vocabulary size |V|.
        trajs: training trajectories (consecutive pairs are the transitions).
        alpha: Laplace smoothing added to every transition count.

    Returns:
        (|V|, |V|) float array of ``log P(j | i)`` (rows sum to 1 in prob space).
    """
    counts = np.full((n_pois, n_pois), alpha, dtype=float)
    for t in trajs:
        seq = t.pois
        for i in range(len(seq) - 1):
            counts[seq[i], seq[i + 1]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    return np.log(counts / row_sums)


def markov_factory(mode: str = "greedy", beam: int = 5, alpha: float = 0.1):
    """Return a factory for the Markov baseline.

    Args:
        mode: ``"greedy"`` (Chen "Markov") or ``"beam"`` (Chen "MarkovPath").
        beam: beam width when ``mode == "beam"``.
        alpha: Laplace smoothing for the transition counts.
    """

    def factory(city: FlickrCity, train: List[Trajectory]) -> MarkovRecommender:
        logtrans = fit_log_transition(city.n_pois, train, alpha=alpha)
        return MarkovRecommender(logtrans, mode=mode, beam=beam)

    return factory


#: Registry of the classical baselines (name -> factory builder), used by the
#: run harness and the Colab notebook. Each maps to a Chen 2016 method.
def classical_factories() -> Dict[str, object]:
    """Return ``{display_name: factory}`` for the classical baselines.

    Returns:
        Ordered dict-like mapping from the baseline name (matching the Chen 2016
        column it reproduces) to its :data:`RecommenderFactory`.
    """
    return {
        "Random": random_factory(seed=0),
        "PoiPopularity": popularity_factory(),
        "Markov": markov_factory(mode="greedy"),
        "MarkovPath": markov_factory(mode="beam", beam=5),
    }
