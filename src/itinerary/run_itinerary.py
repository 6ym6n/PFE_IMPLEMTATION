"""Entry point: decode itineraries from the frozen Phase-1 checkpoint.

Loads ``best.pt`` (unmodified next-POI model) + processed data for a city,
builds one query per test session, decodes greedy and beam routes, and writes
itinerary metrics over all sessions and the length>=3 subset. No training.
Mirrors the load/save plumbing of src.train.train_model.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from src.itinerary.eval_itinerary import evaluate_itinerary, fmt_itinerary_metrics
from src.itinerary.query import build_eval_queries
from src.models.next_poi import NextPOIModel


def estimate_assumed_dt_hours(train_df: pd.DataFrame) -> float:
    """Median in-session Δt (hours) from training data — a documented decode constant.

    Args:
        train_df: training DataFrame with a ``delta_t`` column (hours).

    Returns:
        Median positive Δt, or 1.0 if unavailable. Used for the context features
        at decode time (length mode; does not affect the stop rule).
    """
    if "delta_t" not in train_df.columns:
        return 1.0
    pos = train_df.loc[train_df["delta_t"] > 0, "delta_t"]
    return float(pos.median()) if len(pos) else 1.0


def run_itinerary(
    name: str,
    project_root: str,
    device: torch.device,
    fixed_end: bool = True,
    beam: int = 3,
    d_p: int = 128,
    d_u: int = 64,
    d_c: int = 32,
    d_h: int = 128,
    d_hidden: int = 256,
    dropout: float = 0.2,
) -> Dict[str, Any]:
    """Run greedy + beam itinerary decoding for one city and persist results.

    Reads from ``{project_root}/data/processed/{name}/`` and
    ``{project_root}/checkpoints/{name}/best.pt``; writes to
    ``{project_root}/results/{name}_itinerary_{greedy,beam}.json``.

    Args:
        name: city name (e.g. ``"NYC"``).
        project_root: root dir with ``data/``, ``checkpoints/``, ``results/``.
        device: torch device.
        fixed_end: fix each query's end POI to the session's last POI.
        beam: beam width for the beam decoder.
        d_p, d_u, d_c, d_h, d_hidden, dropout: model dims (must match training).

    Returns:
        Dict mapping ``"greedy"``/``"beam"`` → {``"all"``: metrics, ``"len_ge_3"``: metrics}.
    """
    proc = os.path.join(project_root, "data/processed", name)
    test_df = pd.read_parquet(os.path.join(proc, "test.parquet"))
    train_df = pd.read_parquet(os.path.join(proc, "train.parquet"))
    poi_coords = np.load(os.path.join(proc, "poi_coords.npy"))
    edge_index = torch.load(os.path.join(proc, "edge_index.pt")).to(device)
    with open(os.path.join(proc, "meta.json")) as f:
        meta = json.load(f)

    model = NextPOIModel(
        n_pois=meta["n_pois"], n_users=meta["n_users"],
        d_p=d_p, d_u=d_u, d_c=d_c, d_h=d_h, d_hidden=d_hidden, dropout=dropout,
    ).to(device)
    ckpt = os.path.join(project_root, "checkpoints", name, "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    assumed_dt = estimate_assumed_dt_hours(train_df)
    queries = build_eval_queries(test_df, fixed_end=fixed_end, min_len=2)
    print(
        f"[{name}] {len(queries)} itinerary queries "
        f"(length>=3: {sum(1 for q in queries if q.K >= 3)}) | "
        f"assumed_dt_hours={assumed_dt:.3f}"
    )

    results: Dict[str, Any] = {}
    for decoder in ("greedy", "beam"):
        all_m = evaluate_itinerary(
            model, queries, edge_index, poi_coords, device,
            decoder=decoder, beam=beam, assumed_dt_hours=assumed_dt, min_len=1,
        )
        ge3_m = evaluate_itinerary(
            model, queries, edge_index, poi_coords, device,
            decoder=decoder, beam=beam, assumed_dt_hours=assumed_dt, min_len=3,
        )
        print(f"[{name}] {decoder:6s} | ALL      : {fmt_itinerary_metrics(all_m)}")
        print(f"[{name}] {decoder:6s} | len>=3   : {fmt_itinerary_metrics(ge3_m)}")
        results[decoder] = {"all": all_m, "len_ge_3": ge3_m}

        out = os.path.join(project_root, "results", f"{name}_itinerary_{decoder}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {"decoder": decoder, "fixed_end": fixed_end, "beam": beam,
                 "assumed_dt_hours": assumed_dt, "all": all_m, "len_ge_3": ge3_m},
                f, indent=2,
            )

    return results
