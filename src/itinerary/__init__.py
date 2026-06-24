"""Itinerary recommendation (Phase 2): decode the frozen next-POI model into routes.

Inference-only layer over the Phase-1 NextPOIModel. See itinerary_plan.md.

Public API:
    ItineraryQuery, build_eval_queries   — src.itinerary.query
    rollout_greedy, rollout_beam         — src.itinerary.decode
    pairs_f1, set_f1, evaluate_itinerary — src.itinerary.eval_itinerary
    run_itinerary                        — src.itinerary.run_itinerary
"""

from src.itinerary.decode import rollout_beam, rollout_greedy
from src.itinerary.eval_itinerary import (
    evaluate_itinerary,
    fmt_itinerary_metrics,
    pairs_f1,
    set_f1,
)
from src.itinerary.pointer_model import (
    PointerItineraryModel,
    evaluate_pointer,
    pointer_rollout_beam,
    pointer_rollout_greedy,
)
from src.itinerary.query import ItineraryQuery, build_eval_queries
from src.itinerary.seq_dataset import ItinerarySeqDataset, seq_collate_fn

__all__ = [
    # Strategy A (frozen rollout)
    "ItineraryQuery",
    "build_eval_queries",
    "rollout_greedy",
    "rollout_beam",
    "pairs_f1",
    "set_f1",
    "evaluate_itinerary",
    "fmt_itinerary_metrics",
    # Strategy B (pointer network)
    "PointerItineraryModel",
    "pointer_rollout_greedy",
    "pointer_rollout_beam",
    "evaluate_pointer",
    "ItinerarySeqDataset",
    "seq_collate_fn",
]
