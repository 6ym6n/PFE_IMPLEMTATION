"""Published F1 / pairs-F1 on the Flickr trip-recommendation datasets.

Hand-curated from the downloaded papers (see ``ARTICLES/`` and the extraction in
``results/flickr/published_literature_raw.json``). Only the **directly
comparable** numbers are kept here: methods evaluated with the *canonical
protocol* — leave-one-out cross-validation, the first and last POI given as the
query, trajectories of length >= 3, and pairs-F1 as defined by Chen 2016 — all on
the 0-1 scale. This is the protocol our :mod:`src.flickr` harness reproduces, so
these numbers sit on the SAME scale as ours.

Sources (all on the 0-1 scale, leave-one-out, endpoints given, length>=3):
  * Chen2016   — Chen, Ong, Xie, "Learning Points and Routes to Recommend
                 Trajectories", CIKM 2016 (Tables 3 F1, 4 pairs-F1). THE source
                 of pairs-F1 and our data (its ``traj-*.csv`` files). All five
                 cities; every classical baseline.
  * DeepTrip2019  — Gao et al., SIGSPATIAL 2019 (Table 2). Edinburgh/Glasgow/Osaka.
  * SelfTrip2022  — Gao et al., KBS 2022 (Table 2). Edin/Glas/Osaka/Toronto.
  * ARTrip2024    — Shu et al., SIGIR 2024. Supplies CTLTR and Bert-Trip pairs-F1
                    (CTLTR's own paper is paywalled) plus AR-Trip (recent SOTA).

NOT included here because they use a DIFFERENT protocol (so NOT comparable):
  * POIBERT2022 (Ho & Lim) — 80/20 chronological split, percentage F1, no pairs-F1.
  * DLIR2025 (Halder et al.) — 8-hour session split + >=3-checkin filter, a
    different "F1" (~0.49); not the Chen pairs-F1 protocol.
  * TourEmbedding2021 (Ho & Lim) — precision/recall/F1 only, no pairs-F1.
See ``NON_COMPARABLE_NOTES`` for the specifics.

This module is pure data + helpers; it imports nothing heavy and has no state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Pub:
    """One published (method, city) result on the comparable protocol.

    Attributes:
        method: method name as in the source paper.
        city: full city name.
        F1: point/set F1 (0-1), or None if the paper did not report it.
        pairs_F1: order-aware pairs-F1 (0-1), or None if not reported.
        source: citation key (e.g. ``"Chen2016"``).
        family: one of ``"naive"``, ``"classical"``, ``"OR"``, ``"neural"``.
    """

    method: str
    city: str
    F1: Optional[float]
    pairs_F1: Optional[float]
    source: str
    family: str


#: Every directly-comparable published result (0-1, LOO, endpoints given, len>=3).
PUBLISHED: List[Pub] = [
    Pub('PoiPopularity', 'Osaka', F1=0.663, pairs_F1=0.365, source='Chen2016', family='naive'),
    Pub('Random', 'Osaka', F1=0.621, pairs_F1=0.304, source='Chen2016', family='naive'),
    Pub('Markov', 'Osaka', F1=0.697, pairs_F1=0.445, source='Chen2016', family='classical'),
    Pub('MarkovPath', 'Osaka', F1=0.706, pairs_F1=0.442, source='Chen2016', family='classical'),
    Pub('PoiRank', 'Osaka', F1=0.745, pairs_F1=0.511, source='Chen2016', family='classical'),
    Pub('Rank+Markov', 'Osaka', F1=0.715, pairs_F1=0.486, source='Chen2016', family='classical'),
    Pub('Rank+MarkovPath', 'Osaka', F1=0.732, pairs_F1=0.489, source='Chen2016', family='classical'),
    Pub('PersTour', 'Osaka', F1=0.686, pairs_F1=0.468, source='Chen2016', family='OR'),
    Pub('PersTour-L', 'Osaka', F1=0.686, pairs_F1=0.406, source='Chen2016', family='OR'),
    Pub('AR-Trip', 'Osaka', F1=0.871, pairs_F1=0.828, source='ARTrip2024', family='neural'),
    Pub('Bert-Trip', 'Osaka', F1=0.854, pairs_F1=0.74, source='ARTrip2024', family='neural'),
    Pub('CTLTR', 'Osaka', F1=0.802, pairs_F1=0.719, source='ARTrip2024', family='neural'),
    Pub('DeepTrip', 'Osaka', F1=0.834, pairs_F1=0.755, source='DeepTrip2019', family='neural'),
    Pub('SelfTrip', 'Osaka', F1=0.857, pairs_F1=0.851, source='SelfTrip2022', family='neural'),
    Pub('PoiPopularity', 'Glasgow', F1=0.745, pairs_F1=0.507, source='Chen2016', family='naive'),
    Pub('Random', 'Glasgow', F1=0.632, pairs_F1=0.32, source='Chen2016', family='naive'),
    Pub('Markov', 'Glasgow', F1=0.725, pairs_F1=0.495, source='Chen2016', family='classical'),
    Pub('MarkovPath', 'Glasgow', F1=0.732, pairs_F1=0.485, source='Chen2016', family='classical'),
    Pub('PoiRank', 'Glasgow', F1=0.768, pairs_F1=0.548, source='Chen2016', family='classical'),
    Pub('Rank+Markov', 'Glasgow', F1=0.754, pairs_F1=0.545, source='Chen2016', family='classical'),
    Pub('Rank+MarkovPath', 'Glasgow', F1=0.762, pairs_F1=0.533, source='Chen2016', family='classical'),
    Pub('PersTour', 'Glasgow', F1=0.801, pairs_F1=0.643, source='Chen2016', family='OR'),
    Pub('PersTour-L', 'Glasgow', F1=0.66, pairs_F1=0.352, source='Chen2016', family='OR'),
    Pub('AR-Trip', 'Glasgow', F1=0.881, pairs_F1=0.82, source='ARTrip2024', family='neural'),
    Pub('Bert-Trip', 'Glasgow', F1=0.849, pairs_F1=0.74, source='ARTrip2024', family='neural'),
    Pub('CTLTR', 'Glasgow', F1=0.838, pairs_F1=0.763, source='ARTrip2024', family='neural'),
    Pub('DeepTrip', 'Glasgow', F1=0.831, pairs_F1=0.782, source='DeepTrip2019', family='neural'),
    Pub('SelfTrip', 'Glasgow', F1=0.855, pairs_F1=0.818, source='SelfTrip2022', family='neural'),
    Pub('PoiPopularity', 'Edinburgh', F1=0.701, pairs_F1=0.436, source='Chen2016', family='naive'),
    Pub('Random', 'Edinburgh', F1=0.57, pairs_F1=0.261, source='Chen2016', family='naive'),
    Pub('Markov', 'Edinburgh', F1=0.645, pairs_F1=0.417, source='Chen2016', family='classical'),
    Pub('MarkovPath', 'Edinburgh', F1=0.678, pairs_F1=0.4, source='Chen2016', family='classical'),
    Pub('PoiRank', 'Edinburgh', F1=0.7, pairs_F1=0.432, source='Chen2016', family='classical'),
    Pub('Rank+Markov', 'Edinburgh', F1=0.659, pairs_F1=0.444, source='Chen2016', family='classical'),
    Pub('Rank+MarkovPath', 'Edinburgh', F1=0.697, pairs_F1=0.428, source='Chen2016', family='classical'),
    Pub('PersTour', 'Edinburgh', F1=0.656, pairs_F1=0.417, source='Chen2016', family='OR'),
    Pub('PersTour-L', 'Edinburgh', F1=0.651, pairs_F1=0.359, source='Chen2016', family='OR'),
    Pub('AR-Trip', 'Edinburgh', F1=0.868, pairs_F1=0.808, source='ARTrip2024', family='neural'),
    Pub('Bert-Trip', 'Edinburgh', F1=0.814, pairs_F1=0.672, source='ARTrip2024', family='neural'),
    Pub('CTLTR', 'Edinburgh', F1=0.728, pairs_F1=0.681, source='ARTrip2024', family='neural'),
    Pub('DeepTrip', 'Edinburgh', F1=0.765, pairs_F1=0.66, source='DeepTrip2019', family='neural'),
    Pub('SelfTrip', 'Edinburgh', F1=0.783, pairs_F1=0.779, source='SelfTrip2022', family='neural'),
    Pub('PoiPopularity', 'Toronto', F1=0.678, pairs_F1=0.384, source='Chen2016', family='naive'),
    Pub('Random', 'Toronto', F1=0.621, pairs_F1=0.31, source='Chen2016', family='naive'),
    Pub('Markov', 'Toronto', F1=0.669, pairs_F1=0.407, source='Chen2016', family='classical'),
    Pub('MarkovPath', 'Toronto', F1=0.688, pairs_F1=0.405, source='Chen2016', family='classical'),
    Pub('PoiRank', 'Toronto', F1=0.754, pairs_F1=0.518, source='Chen2016', family='classical'),
    Pub('Rank+Markov', 'Toronto', F1=0.723, pairs_F1=0.512, source='Chen2016', family='classical'),
    Pub('Rank+MarkovPath', 'Toronto', F1=0.751, pairs_F1=0.514, source='Chen2016', family='classical'),
    Pub('PersTour', 'Toronto', F1=0.72, pairs_F1=0.504, source='Chen2016', family='OR'),
    Pub('PersTour-L', 'Toronto', F1=0.643, pairs_F1=0.333, source='Chen2016', family='OR'),
    Pub('AR-Trip', 'Toronto', F1=0.892, pairs_F1=0.839, source='ARTrip2024', family='neural'),
    Pub('Bert-Trip', 'Toronto', F1=0.859, pairs_F1=0.733, source='ARTrip2024', family='neural'),
    Pub('CTLTR', 'Toronto', F1=0.806, pairs_F1=0.748, source='ARTrip2024', family='neural'),
    Pub('SelfTrip', 'Toronto', F1=0.851, pairs_F1=0.835, source='SelfTrip2022', family='neural'),
    Pub('PoiPopularity', 'Melbourne', F1=0.62, pairs_F1=0.316, source='Chen2016', family='naive'),
    Pub('Random', 'Melbourne', F1=0.558, pairs_F1=0.248, source='Chen2016', family='naive'),
    Pub('Markov', 'Melbourne', F1=0.577, pairs_F1=0.288, source='Chen2016', family='classical'),
    Pub('MarkovPath', 'Melbourne', F1=0.595, pairs_F1=0.294, source='Chen2016', family='classical'),
    Pub('PoiRank', 'Melbourne', F1=0.637, pairs_F1=0.339, source='Chen2016', family='classical'),
    Pub('Rank+Markov', 'Melbourne', F1=0.613, pairs_F1=0.351, source='Chen2016', family='classical'),
    Pub('Rank+MarkovPath', 'Melbourne', F1=0.639, pairs_F1=0.344, source='Chen2016', family='classical'),
    Pub('PersTour', 'Melbourne', F1=0.483, pairs_F1=0.216, source='Chen2016', family='OR'),
    Pub('PersTour-L', 'Melbourne', F1=0.576, pairs_F1=0.266, source='Chen2016', family='OR'),
]

#: Protocol caveats for the papers we deliberately exclude from the headline table.
NON_COMPARABLE_NOTES: Dict[str, str] = {
    "POIBERT2022": "80/20 chronological split (not leave-one-out); F1 reported as "
                   "a percentage; no pairs-F1. Not comparable.",
    "DLIR2025": "8-hour session split + >=3-check-in POI/user filter; reports a "
                "different F1 (~0.49 range); no Chen pairs-F1. Not comparable.",
    "TourEmbedding2021": "Precision/Recall/F1 of predicted POIs only; no pairs-F1; "
                         "different query setup. Not comparable.",
}


def published_for_city(city: str, metric: str = "pairs_F1") -> Dict[str, float]:
    """All comparable published values for one city, keyed by method.

    Args:
        city: full city name (e.g. ``"Edinburgh"``).
        metric: ``"pairs_F1"`` or ``"F1"``.

    Returns:
        ``{method: value}`` for every comparable result that reports ``metric``
        on ``city`` (skips None). When a method appears from several sources the
        first in :data:`PUBLISHED` wins (canonical source order).
    """
    out: Dict[str, float] = {}
    for p in PUBLISHED:
        if p.city != city:
            continue
        val = p.pairs_F1 if metric == "pairs_F1" else p.F1
        if val is not None and p.method not in out:
            out[p.method] = val
    return out


def pairs_f1_range(city: str) -> Optional[tuple]:
    """(min, max) comparable published pairs-F1 for a city, or None if empty."""
    vals = list(published_for_city(city, "pairs_F1").values())
    return (min(vals), max(vals)) if vals else None
