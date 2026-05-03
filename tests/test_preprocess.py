"""Shape and behavior tests for preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocess import (
    build_sessions,
    chronological_split,
    haversine_km,
    iterative_filter,
    reindex,
)


def test_haversine_zero_distance() -> None:
    d = haversine_km(np.array([40.0]), np.array([-73.0]),
                     np.array([40.0]), np.array([-73.0]))
    assert float(d[0]) < 1e-6


def test_haversine_known_distance() -> None:
    """NYC (40.7128, -74.0060) → LA (34.0522, -118.2437) ≈ 3936 km."""
    d = haversine_km(np.array([40.7128]), np.array([-74.0060]),
                     np.array([34.0522]), np.array([-118.2437]))
    assert 3900 < float(d[0]) < 4000


def test_iterative_filter_drops_rare() -> None:
    rows = []
    # 5 frequent users × 12 popular POIs = enough to survive a min=10 filter
    for u in range(5):
        for p in range(12):
            rows.append({"user_id": f"u{u}", "poi_id": f"p{p}"})
    # one rare user with 2 visits to a popular POI — should be dropped
    rows.append({"user_id": "rare", "poi_id": "p0"})
    rows.append({"user_id": "rare", "poi_id": "p1"})
    df = pd.DataFrame(rows)
    out = iterative_filter(df, min_user=10, min_poi=2)
    assert "rare" not in set(out["user_id"])


def test_reindex_contiguous() -> None:
    df = pd.DataFrame({"user_id": ["b", "a", "c", "a"],
                       "poi_id": ["p2", "p1", "p1", "p3"]})
    out, u2i, p2i = reindex(df)
    assert sorted(u2i.values()) == [0, 1, 2]
    assert sorted(p2i.values()) == [0, 1, 2]
    assert out["user_idx"].max() == len(u2i) - 1
    assert out["poi_idx"].max() == len(p2i) - 1


def test_build_sessions_splits_at_24h_gap() -> None:
    """Two consecutive check-ins 30h apart must end up in different sessions."""
    df = pd.DataFrame({
        "user_id": ["u1"] * 4,
        "user_idx": [0] * 4,
        "poi_id": ["p1", "p2", "p3", "p4"],
        "poi_idx": [0, 1, 2, 3],
        "lat":  [40.7, 40.71, 40.72, 40.73],
        "lon":  [-74.0, -74.01, -74.02, -74.03],
        "timestamp": [
            pd.Timestamp("2023-01-01 10:00"),
            pd.Timestamp("2023-01-01 11:00"),  # 1h gap → same session
            pd.Timestamp("2023-01-02 18:00"),  # 31h gap → new session
            pd.Timestamp("2023-01-02 19:00"),  # 1h gap → same session
        ],
    })
    out = build_sessions(df, gap_hours=24)
    # Both sessions have length 2 → both kept
    assert out["session_id"].nunique() == 2
    # First step in each session has delta_t = 0 by convention
    firsts = out.groupby("session_id").first()
    assert (firsts["delta_t"] == 0.0).all()
    assert (firsts["delta_d"] == 0.0).all()


def test_build_sessions_drops_singletons() -> None:
    df = pd.DataFrame({
        "user_id": ["u1"] * 2,
        "user_idx": [0] * 2,
        "poi_id": ["p1", "p2"],
        "poi_idx": [0, 1],
        "lat": [40.7, 40.8],
        "lon": [-74.0, -74.1],
        "timestamp": [
            pd.Timestamp("2023-01-01 10:00"),
            pd.Timestamp("2023-01-05 10:00"),  # 4-day gap → both isolated
        ],
    })
    out = build_sessions(df, gap_hours=24)
    assert len(out) == 0


def test_chronological_split_no_user_overlap_in_time() -> None:
    """Within a user, train timestamps must not exceed val/test timestamps."""
    rows = []
    for u in range(2):
        for s in range(10):
            t0 = pd.Timestamp("2023-01-01") + pd.Timedelta(days=s)
            for k in range(3):
                rows.append({
                    "user_idx": u,
                    "session_id": u * 100 + s,
                    "poi_idx": k,
                    "timestamp": t0 + pd.Timedelta(minutes=10 * k),
                })
    df = pd.DataFrame(rows)
    train, val, test = chronological_split(df, train_frac=0.7, val_frac=0.1)
    for u in df["user_idx"].unique():
        t_train = train[train["user_idx"] == u]["timestamp"]
        t_val = val[val["user_idx"] == u]["timestamp"]
        t_test = test[test["user_idx"] == u]["timestamp"]
        if len(t_train) and len(t_val):
            assert t_train.max() <= t_val.min()
        if len(t_val) and len(t_test):
            assert t_val.max() <= t_test.min()
