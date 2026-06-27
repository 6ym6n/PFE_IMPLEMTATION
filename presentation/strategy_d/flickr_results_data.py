"""Single source of truth for the Strategy-D presentation + report.

All numbers here are the measured results of the Flickr trip-recommendation work
(``src/flickr/``) and the curated published comparison (``src/flickr/published.py``).
Both the PDF (``make_pdf.py``) and the PowerPoint (``make_pptx.py``) read from this
module so the deck and the document never disagree.
"""

from __future__ import annotations

CITIES = ["Toronto", "Osaka", "Glasgow", "Edinburgh", "Melbourne"]

#: Per-city dataset statistics (from the canonical Chen 2016 traj-*/poi-* files).
DATASET_STATS = {
    #            POIs  users  #traj   #traj>=3 (the evaluated set)
    "Toronto":   (29, 1395, 6057, 335),
    "Osaka":     (27,  450, 1115,  47),
    "Glasgow":   (27,  601, 2227, 112),
    "Edinburgh": (28, 1454, 5028, 634),
    "Melbourne": (88, 1000, 5106, 442),
}

#: Our measured classical pairs-F1 (leave-one-out, length>=3, train_pool=ge3).
OURS_PAIRS_F1 = {
    "Random":        {"Toronto": 0.298, "Osaka": 0.301, "Glasgow": 0.301, "Edinburgh": 0.270, "Melbourne": 0.227},
    "PoiPopularity": {"Toronto": 0.443, "Osaka": 0.413, "Glasgow": 0.510, "Edinburgh": 0.439, "Melbourne": 0.320},
    "Markov":        {"Toronto": 0.504, "Osaka": 0.421, "Glasgow": 0.587, "Edinburgh": 0.449, "Melbourne": 0.333},
    "MarkovPath":    {"Toronto": 0.528, "Osaka": 0.398, "Glasgow": 0.543, "Edinburgh": 0.452, "Melbourne": 0.346},
}

#: Our measured classical point-F1 (same protocol).
OURS_F1 = {
    "Random":        {"Toronto": 0.611, "Osaka": 0.613, "Glasgow": 0.620, "Edinburgh": 0.581, "Melbourne": 0.534},
    "PoiPopularity": {"Toronto": 0.714, "Osaka": 0.698, "Glasgow": 0.747, "Edinburgh": 0.704, "Melbourne": 0.619},
    "Markov":        {"Toronto": 0.737, "Osaka": 0.685, "Glasgow": 0.786, "Edinburgh": 0.702, "Melbourne": 0.619},
    "MarkovPath":    {"Toronto": 0.758, "Osaka": 0.681, "Glasgow": 0.762, "Edinburgh": 0.705, "Melbourne": 0.628},
}

#: Our measured learned-model pairs-F1 (GCN + pointer, 60 epochs/fold, Colab GPU).
POINTER_PAIRS_F1 = {
    "Pointer (beam)":   {"Toronto": 0.431, "Osaka": 0.399, "Glasgow": 0.489, "Edinburgh": 0.414, "Melbourne": 0.312},
    "Pointer (greedy)": {"Toronto": 0.430, "Osaka": 0.362, "Glasgow": 0.486, "Edinburgh": 0.414, "Melbourne": 0.309},
}
POINTER_F1 = {
    "Pointer (beam)": {"Toronto": 0.705, "Osaka": 0.684, "Glasgow": 0.734, "Edinburgh": 0.689, "Melbourne": 0.614},
}

#: Chen 2016's published pairs-F1 (the faithfulness-check target).
CHEN_PAIRS_F1 = {
    "Random":        {"Toronto": 0.310, "Osaka": 0.304, "Glasgow": 0.320, "Edinburgh": 0.261, "Melbourne": 0.248},
    "PoiPopularity": {"Toronto": 0.384, "Osaka": 0.365, "Glasgow": 0.507, "Edinburgh": 0.436, "Melbourne": 0.316},
    "Markov":        {"Toronto": 0.407, "Osaka": 0.445, "Glasgow": 0.495, "Edinburgh": 0.417, "Melbourne": 0.288},
}

#: Published comparison (directly-comparable subset; pairs-F1, 0-1 scale).
#: Cities Toronto/Osaka/Glasgow/Edinburgh (Melbourne is reported mostly by the classical line).
PUBLISHED_PAIRS_F1 = {
    # method:                (Toronto, Osaka, Glasgow, Edinburgh), family, year
    "PoiRank (Chen)":        ((0.518, 0.511, 0.548, 0.432), "classical", 2016),
    "Rank+Markov (Chen)":    ((0.512, 0.486, 0.545, 0.444), "classical", 2016),
    "DeepTrip":              ((0.748, 0.755, 0.782, 0.660), "neural", 2019),
    "CTLTR":                 ((0.748, 0.719, 0.763, 0.681), "neural", 2021),
    "SelfTrip":              ((0.835, 0.851, 0.818, 0.779), "neural", 2022),
    "AR-Trip":               ((0.839, 0.828, 0.820, 0.808), "neural", 2024),
}
PUB_CITIES = ["Toronto", "Osaka", "Glasgow", "Edinburgh"]

#: Worked pairs-F1 example used in both deliverables.
PAIRS_EXAMPLE = {
    "truth": ["A", "B", "C", "D"],
    "pred":  ["A", "C", "B", "D"],
    "truth_pairs": ["(A,B)", "(A,C)", "(A,D)", "(B,C)", "(B,D)", "(C,D)"],
    "pred_pairs":  ["(A,C)", "(A,B)", "(A,D)", "(C,B)", "(C,D)", "(B,D)"],
    "shared": ["(A,B)", "(A,C)", "(A,D)", "(C,D)", "(B,D)"],   # same order in both
    "value": 5.0 / 6.0,
}

#: Headline ranges (pairs-F1).
NYC_RANGE = (0.26, 0.29)        # Phase-2 itinerary on Foursquare NYC (not comparable)
FLICKR_OURS_RANGE = (0.23, 0.59)
PUBLISHED_RANGE = (0.26, 0.85)

#: One concrete (illustrative) Edinburgh-style trajectory for the running example.
EXAMPLE_TRAJECTORY = ["Castle", "Royal Mile", "Holyrood Palace", "Arthur's Seat", "National Museum"]


# ===========================================================================
# Thesis-wide numbers (Phase 1 + Phase 2 NYC A-vs-B), for the full-thesis deck.
# ===========================================================================
#: Phase 1 — next-POI (the personalized, context-aware engine) on Foursquare NYC, full vocab.
PHASE1_METRICS = {
    "HR@1": 0.1874, "HR@5": 0.4760, "HR@10": 0.5879,
    "NDCG@5": 0.3386, "NDCG@10": 0.3751, "MRR": 0.3163,
}

#: HR@1 tier from the LLM4POI benchmark table (ours bolded in the deck).
PHASE1_TIER = [
    ("LSTM", 0.130), ("STGCN", 0.180), ("Ours (GCN+GRU+user)", 0.187),
    ("STAN", 0.220), ("GETNext", 0.240), ("STHGCN", 0.270), ("LLM4POI", 0.340),
]

#: Phase 2 NYC itinerary — the integration experiment (length>=3 test, n=2,880).
#: Strategy A = decoupled decoding of the next-POI engine; Strategy B = integrated pointer.
AVSB_NYC = {
    # method:                          pairs-F1, set-F1, exact, kind
    "A — frozen rollout (greedy)":     (0.2887, 0.609, 0.054, "decoupled"),
    "A — frozen rollout (beam 3)":     (0.2902, 0.610, 0.057, "decoupled"),
    "B-v1 — pointer (no context)":     (0.2585, 0.578, 0.043, "integrated"),
    "B-v2 — pointer (+ context)":      (0.2610, 0.000, 0.000, "integrated"),  # beam3; set-F1/exact not re-logged
}
AVSB_N = 2880
