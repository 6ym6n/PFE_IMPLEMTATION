"""Shape and length tests for POISessionDataset and collate_fn."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset import POISessionDataset, collate_fn


def _toy_df() -> pd.DataFrame:
    """Two sessions: lengths 4 and 3 → 3 + 2 = 5 examples."""
    rows = [
        # Session 1, user 0, length 4
        {"session_id": 1, "user_idx": 0, "timestamp": pd.Timestamp("2023-01-01 10:00"),
         "poi_idx": 0, "delta_d": 0.0, "delta_t": 0.0},
        {"session_id": 1, "user_idx": 0, "timestamp": pd.Timestamp("2023-01-01 10:30"),
         "poi_idx": 1, "delta_d": 0.5, "delta_t": 0.5},
        {"session_id": 1, "user_idx": 0, "timestamp": pd.Timestamp("2023-01-01 11:00"),
         "poi_idx": 2, "delta_d": 0.7, "delta_t": 0.5},
        {"session_id": 1, "user_idx": 0, "timestamp": pd.Timestamp("2023-01-01 11:30"),
         "poi_idx": 3, "delta_d": 0.3, "delta_t": 0.5},
        # Session 2, user 1, length 3
        {"session_id": 2, "user_idx": 1, "timestamp": pd.Timestamp("2023-01-02 09:00"),
         "poi_idx": 4, "delta_d": 0.0, "delta_t": 0.0},
        {"session_id": 2, "user_idx": 1, "timestamp": pd.Timestamp("2023-01-02 09:15"),
         "poi_idx": 5, "delta_d": 1.2, "delta_t": 0.25},
        {"session_id": 2, "user_idx": 1, "timestamp": pd.Timestamp("2023-01-02 09:45"),
         "poi_idx": 6, "delta_d": 0.4, "delta_t": 0.5},
    ]
    return pd.DataFrame(rows)


def test_dataset_example_count() -> None:
    df = _toy_df()
    ds = POISessionDataset(df, max_seq_len=100)
    # Session of length L → L-1 examples; total = 3 + 2 = 5
    assert len(ds) == 5


def test_dataset_example_shapes_and_targets() -> None:
    ds = POISessionDataset(_toy_df())
    # First example from session 1: history = [0], target = 1
    e0 = ds[0]
    assert list(e0["history_pois"]) == [0]
    assert e0["target"] == 1
    assert e0["user"] == 0
    # Second example from session 1: history = [0, 1], target = 2
    e1 = ds[1]
    assert list(e1["history_pois"]) == [0, 1]
    assert e1["target"] == 2


def test_dataset_max_seq_len_truncates_left() -> None:
    """Build a long session and confirm truncation keeps the most recent steps."""
    n = 50
    df = pd.DataFrame({
        "session_id": [1] * n,
        "user_idx": [0] * n,
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="30min"),
        "poi_idx": np.arange(n),
        "delta_d": np.zeros(n, dtype=np.float32),
        "delta_t": np.zeros(n, dtype=np.float32),
    })
    ds = POISessionDataset(df, max_seq_len=5)
    # Last example: target = 49, history should be [44..48] (last 5)
    last = ds[-1]
    assert last["target"] == 49
    assert list(last["history_pois"]) == [44, 45, 46, 47, 48]


def test_collate_fn_pads_to_batch_max() -> None:
    ds = POISessionDataset(_toy_df())
    batch = [ds[0], ds[1], ds[2]]  # history lengths 1, 2, 3
    out = collate_fn(batch)

    assert out["poi_ids"].shape == (3, 3)
    assert out["delta_d"].shape == (3, 3)
    assert out["delta_t"].shape == (3, 3)
    assert out["user_ids"].shape == (3,)
    assert out["lengths"].tolist() == [1, 2, 3]
    assert out["targets"].shape == (3,)
    # Padding zeros after the real history of the first example (length 1)
    assert out["poi_ids"][0, 1].item() == 0
    assert out["poi_ids"][0, 2].item() == 0


def test_collate_fn_dtypes() -> None:
    import torch
    ds = POISessionDataset(_toy_df())
    out = collate_fn([ds[0], ds[1]])
    assert out["poi_ids"].dtype == torch.long
    assert out["delta_d"].dtype == torch.float
    assert out["delta_t"].dtype == torch.float
    assert out["user_ids"].dtype == torch.long
    assert out["lengths"].dtype == torch.long
    assert out["targets"].dtype == torch.long
