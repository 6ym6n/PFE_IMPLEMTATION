"""Strategy D: itinerary recommendation on the Flickr photo-trajectory datasets.

A self-contained chapter that re-runs the thesis's itinerary task on the small
Flickr "User-POI visits" benchmark (Toronto, Osaka, Glasgow, Edinburgh,
Melbourne) under the *canonical* trip-recommendation protocol — leave-one-out
cross-validation, origin+destination given, trajectories of length >= 3,
pairs-F1 (Chen 2016) — so the numbers are directly comparable to the published
literature (whose pairs-F1 lands in ~0.3-0.85, versus ~0.26-0.29 on Foursquare
NYC where the protocol and data differ).

Public surface:
    data      : FlickrCity, Trajectory, load_city, trajectories_min_len, CITIES
    evaluate  : evaluate_loocv, make_query, RecommenderFactory
    baselines : Random / PoiPopularity / Markov / MarkovPath recommenders + factories
    pointer   : FlickrPointerNet (light GCN+pointer learned recommender, Colab)
    published : PUBLISHED literature numbers (comparable protocol) + helpers
    run_flickr: run_classical_baselines / run_city / format_comparison
"""

from __future__ import annotations

from src.flickr.baselines import (
    MarkovRecommender,
    PopularityRecommender,
    RandomRecommender,
    classical_factories,
    markov_factory,
    popularity_factory,
    random_factory,
)
from src.flickr.data import (
    CITIES,
    CITY_ABBREV,
    FlickrCity,
    Trajectory,
    load_city,
    trajectories_min_len,
)
from src.flickr.evaluate import evaluate_loocv, make_query

__all__ = [
    "CITIES",
    "CITY_ABBREV",
    "FlickrCity",
    "Trajectory",
    "load_city",
    "trajectories_min_len",
    "evaluate_loocv",
    "make_query",
    "RandomRecommender",
    "PopularityRecommender",
    "MarkovRecommender",
    "random_factory",
    "popularity_factory",
    "markov_factory",
    "classical_factories",
]
