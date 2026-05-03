"""Foursquare check-in preprocessing: load → filter → reindex → sessionize → split.

Mirrors Sections 3 (load) and 4 (preprocess) of implementation_guide.md exactly.
All functions are pure: pass DataFrames in, get DataFrames / dicts out.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

#: Foursquare TSMC2014 raw column order.
COLUMNS = [
    "user_id",
    "poi_id",
    "cat_id",
    "cat_name",
    "lat",
    "lon",
    "tz_offset",
    "utc_time",
]


def load_foursquare(path: str) -> pd.DataFrame:
    """Load a Foursquare TSMC2014 raw .txt file into a sorted DataFrame.

    Args:
        path: filesystem path to ``dataset_TSMC2014_NYC.txt`` or TKY.

    Returns:
        DataFrame with columns
        [user_id, poi_id, cat_id, cat_name, lat, lon, timestamp]
        sorted by (user_id, timestamp).
    """
    df = pd.read_csv(path, sep="\t", names=COLUMNS, encoding="latin-1")
    df["timestamp"] = pd.to_datetime(
        df["utc_time"], format="%a %b %d %H:%M:%S +0000 %Y"
    )
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df[["user_id", "poi_id", "cat_id", "cat_name", "lat", "lon", "timestamp"]]


def iterative_filter(
    df: pd.DataFrame,
    min_user: int = 10,
    min_poi: int = 10,
    max_iter: int = 10,
) -> pd.DataFrame:
    """Iteratively drop low-activity users and rare POIs until stable.

    Standard k-core-like filter for next-POI benchmarks.

    Args:
        df: check-in DataFrame with ``user_id`` and ``poi_id`` columns.
        min_user: minimum number of check-ins required per user.
        min_poi:  minimum number of distinct visits required per POI.
        max_iter: hard cap on iterations (usually converges in 2-3).

    Returns:
        Filtered DataFrame with reset index. Same column schema as input.
    """
    for _ in range(max_iter):
        n0 = len(df)
        uc = df["user_id"].value_counts()
        df = df[df["user_id"].isin(uc[uc >= min_user].index)]
        pc = df["poi_id"].value_counts()
        df = df[df["poi_id"].isin(pc[pc >= min_poi].index)]
        if len(df) == n0:
            break
    return df.reset_index(drop=True)


def reindex(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
    """Re-map user_id and poi_id to contiguous integers ``0..N-1``.

    Must be done BEFORE the train/val/test split so val and test never
    contain unseen embedding indices.

    Args:
        df: DataFrame with ``user_id`` and ``poi_id`` columns.

    Returns:
        Tuple ``(df, user2idx, poi2idx)``:
            - df: copy with new columns ``user_idx`` (long) and ``poi_idx`` (long).
            - user2idx: dict mapping original user_id → contiguous int.
            - poi2idx:  dict mapping original poi_id → contiguous int.
    """
    user2idx = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
    poi2idx = {p: i for i, p in enumerate(sorted(df["poi_id"].unique()))}
    df = df.copy()
    df["user_idx"] = df["user_id"].map(user2idx)
    df["poi_idx"] = df["poi_id"].map(poi2idx)
    return df, user2idx, poi2idx


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorized haversine great-circle distance in kilometers.

    Args:
        lat1, lon1, lat2, lon2: same-shape float arrays in degrees.

    Returns:
        Float array (same shape) of distances in km. Earth radius = 6371 km.
    """
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_sessions(df: pd.DataFrame, gap_hours: float = 24.0) -> pd.DataFrame:
    """Group consecutive check-ins into sessions and compute Δd, Δt per pair.

    A new session starts whenever the gap from the previous check-in (same user)
    exceeds ``gap_hours``. Δd/Δt at session boundaries are 0 by convention.
    Sessions of length 1 are dropped (no prediction target).

    Args:
        df: DataFrame with columns ``user_idx``, ``timestamp``, ``lat``, ``lon``.
            Must be the output of :func:`reindex` (or otherwise have ``user_idx``).
        gap_hours: temporal gap that splits sessions (default 24h, Foursquare convention).

    Returns:
        DataFrame with added columns:
            - ``session_id`` (int): globally unique session id.
            - ``delta_t`` (float): hours since previous check-in in session, capped 24.
            - ``delta_d`` (float): km from previous POI in session, capped 100.
        Only rows belonging to sessions with length ≥ 2 are kept.
    """
    df = df.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)

    gap = df.groupby("user_idx")["timestamp"].diff().dt.total_seconds() / 3600
    new_session = (gap > gap_hours) | gap.isna()
    df["session_id"] = new_session.cumsum()  # globally unique

    df["delta_t"] = (
        df.groupby("session_id")["timestamp"].diff().dt.total_seconds() / 3600
    )
    df["delta_t"] = df["delta_t"].fillna(0.0).clip(0, 24)

    lat_prev = df.groupby("session_id")["lat"].shift()
    lon_prev = df.groupby("session_id")["lon"].shift()
    df["delta_d"] = haversine_km(df["lat"], df["lon"], lat_prev, lon_prev)
    df["delta_d"] = df["delta_d"].fillna(0.0).clip(0, 100)

    sess_lengths = df.groupby("session_id").size()
    valid = sess_lengths[sess_lengths >= 2].index
    df = df[df["session_id"].isin(valid)].reset_index(drop=True)
    return df


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user chronological split of sessions (no random shuffling).

    Each user contributes the earliest ``train_frac`` of their sessions to train,
    the next ``val_frac`` to val, and the rest to test. Random splits leak
    badly on next-POI tasks because future check-ins inform past predictions.

    Args:
        df: sessionized DataFrame with ``user_idx``, ``session_id``, ``timestamp``.
        train_frac: fraction of each user's sessions assigned to training.
        val_frac: fraction assigned to validation. Test gets the remainder.

    Returns:
        ``(train_df, val_df, test_df)`` — three DataFrames with reset indices.
    """
    train_parts, val_parts, test_parts = [], [], []
    for _user, g in df.groupby("user_idx"):
        sess_order = (
            g.groupby("session_id")["timestamp"].min().sort_values().index.tolist()
        )
        n = len(sess_order)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_sess = set(sess_order[:n_train])
        val_sess = set(sess_order[n_train : n_train + n_val])
        test_sess = set(sess_order[n_train + n_val :])

        train_parts.append(g[g["session_id"].isin(train_sess)])
        val_parts.append(g[g["session_id"].isin(val_sess)])
        test_parts.append(g[g["session_id"].isin(test_sess)])

    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(val_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
    )
