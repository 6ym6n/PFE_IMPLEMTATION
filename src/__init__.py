"""Next-POI recommendation baseline (TLMR-faithful, Foursquare NYC).

The package is dataset-agnostic — the same modules also work on TKY or any
city following the same TSMC2014 schema; only the notebook orchestrates which
city is run.

Public modules:
    src.data.preprocess  — filtering, sessions, splits
    src.data.graph       — hybrid POI graph construction
    src.data.dataset     — POISessionDataset + collate_fn
    src.models.components — POIGraphEncoder, ContextEncoder
    src.models.next_poi  — NextPOIModel
    src.train            — training loop
    src.eval             — HR@k, NDCG@k, MRR
"""

__version__ = "0.1.0"
