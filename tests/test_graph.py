"""Shape tests for hybrid POI graph construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.graph import (
    build_covisit_edges,
    build_hybrid_graph,
    build_knn_edges,
)


def test_build_covisit_edges_threshold() -> None:
    # Same transition 0→1 happens in 3 sessions → above τ=3? threshold is ≥3 so YES.
    rows = []
    for sid in range(3):
        rows.extend([
            {"session_id": sid, "timestamp": pd.Timestamp("2023-01-01 10:00") + pd.Timedelta(seconds=sid),
             "poi_idx": 0},
            {"session_id": sid, "timestamp": pd.Timestamp("2023-01-01 10:30") + pd.Timedelta(seconds=sid),
             "poi_idx": 1},
        ])
    df = pd.DataFrame(rows)
    edges = build_covisit_edges(df, threshold=3)
    assert (0, 1) in edges
    edges_strict = build_covisit_edges(df, threshold=4)
    assert edges_strict == []


def test_build_covisit_excludes_self_loops() -> None:
    df = pd.DataFrame([
        {"session_id": 1, "timestamp": pd.Timestamp("2023-01-01 10:00"), "poi_idx": 5},
        {"session_id": 1, "timestamp": pd.Timestamp("2023-01-01 10:30"), "poi_idx": 5},
    ])
    edges = build_covisit_edges(df, threshold=1)
    assert edges == []


def test_build_knn_edges_count_and_no_self() -> None:
    """For n POIs and k=2, expect ≤ n*k/2 unique undirected edges (no self)."""
    n_pois = 20
    rng = np.random.default_rng(0)
    coords = rng.uniform(low=[40.0, -74.0], high=[41.0, -73.0], size=(n_pois, 2))
    edges = build_knn_edges(coords, k=2)
    # No self-loops
    for a, b in edges:
        assert a != b
    # Bound: at most n*k unique pairs (way less after dedup)
    assert len(edges) <= n_pois * 2


def test_build_hybrid_graph_returns_pyg_edge_index() -> None:
    n_pois = 10
    rng = np.random.default_rng(0)
    coords = rng.uniform(low=[40.0, -74.0], high=[41.0, -73.0], size=(n_pois, 2))
    train_df = pd.DataFrame([
        {"session_id": 1, "timestamp": pd.Timestamp("2023-01-01 10:00"), "poi_idx": 0},
        {"session_id": 1, "timestamp": pd.Timestamp("2023-01-01 10:30"), "poi_idx": 1},
    ])
    edge_index = build_hybrid_graph(
        train_df, coords, n_pois,
        covisit_threshold=10,  # so co-visit contributes nothing
        knn_k=3,
        verbose=False,
    )
    assert edge_index.dtype == torch.long
    assert edge_index.dim() == 2
    assert edge_index.shape[0] == 2
    # Each undirected edge appears twice → even count
    assert edge_index.shape[1] % 2 == 0
    # All indices in [0, n_pois)
    assert int(edge_index.max().item()) < n_pois
    assert int(edge_index.min().item()) >= 0
