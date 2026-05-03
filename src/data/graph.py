"""Hybrid POI graph construction: co-visit (training only) ∪ geographic kNN.

Mirrors Section 5 of implementation_guide.md exactly.

Edges are built as undirected pairs (smaller index first), de-duplicated across
both sources, then symmetrized into a directed PyG ``edge_index`` of shape (2, E).
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import BallTree


def build_covisit_edges(
    train_df: pd.DataFrame, threshold: int = 3
) -> List[Tuple[int, int]]:
    """Co-visit edges from session-internal POI transitions in training data only.

    For each session in training data, count consecutive transitions
    p_i → p_{i+1}. Aggregate as undirected pairs (smaller index first); keep
    pairs with count ≥ ``threshold``. Self-loops are excluded.

    Args:
        train_df: sessionized training DataFrame with ``session_id``, ``timestamp``,
            ``poi_idx``.
        threshold: minimum number of distinct co-visit transitions required to
            include an edge (default 3).

    Returns:
        List of ``(a, b)`` undirected edges with ``a < b``.
    """
    counts: dict[Tuple[int, int], int] = defaultdict(int)
    for _, g in train_df.groupby("session_id"):
        seq = g.sort_values("timestamp")["poi_idx"].values
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a != b:
                key = (min(a, b), max(a, b))
                counts[key] += 1
    return [edge for edge, c in counts.items() if c >= threshold]


def build_knn_edges(
    poi_coords: np.ndarray, k: int = 10
) -> List[Tuple[int, int]]:
    """Geographic kNN edges: each POI connected to its k nearest by haversine.

    Uses ``sklearn.neighbors.BallTree`` with the haversine metric (expects
    radians). Self-edges are excluded.

    Args:
        poi_coords: (n_pois, 2) float array of [lat_deg, lon_deg].
        k: number of nearest neighbors per POI (default 10).

    Returns:
        List of ``(a, b)`` undirected edges with ``a < b``.
    """
    coords_rad = np.radians(poi_coords)
    tree = BallTree(coords_rad, metric="haversine")
    # k+1 because each POI is its own nearest neighbor.
    _, idx = tree.query(coords_rad, k=k + 1)
    edges: set[Tuple[int, int]] = set()
    for i, neighbors in enumerate(idx):
        for j in neighbors[1:]:  # skip self at position 0
            edges.add((min(i, int(j)), max(i, int(j))))
    return list(edges)


def build_hybrid_graph(
    train_df: pd.DataFrame,
    poi_coords: np.ndarray,
    n_pois: int,
    covisit_threshold: int = 3,
    knn_k: int = 10,
    verbose: bool = True,
) -> torch.Tensor:
    """Construct the hybrid POI graph (co-visit ∪ kNN) and return PyG edge_index.

    Args:
        train_df: training DataFrame for co-visit edges (no leakage from val/test).
        poi_coords: (n_pois, 2) float array of [lat, lon] in degrees.
        n_pois: vocabulary size |V|; used only for the degree sanity check.
        covisit_threshold: τ_cov, minimum co-visit count (default 3).
        knn_k: K, geographic neighbors per POI (default 10).
        verbose: print edge counts and degree summary.

    Returns:
        ``edge_index`` long tensor of shape (2, 2 * |E_unique|): each undirected
        edge appears twice (a→b and b→a) so PyG message passing is symmetric.
    """
    cov = build_covisit_edges(train_df, threshold=covisit_threshold)
    knn = build_knn_edges(poi_coords, k=knn_k)
    all_edges = list(set(cov) | set(knn))
    if verbose:
        print(f"  Co-visit edges: {len(cov):,}")
        print(f"  kNN edges:      {len(knn):,}")
        print(f"  Union:          {len(all_edges):,}")

    src: List[int] = []
    dst: List[int] = []
    for a, b in all_edges:
        src.extend([a, b])
        dst.extend([b, a])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    if verbose:
        degree = torch.zeros(n_pois, dtype=torch.long)
        for s in src:
            degree[s] += 1
        print(
            f"  Degree: min={degree.min().item()}, max={degree.max().item()}, "
            f"mean={degree.float().mean().item():.1f}, "
            f"median={degree.median().item()}"
        )
        n_iso = int((degree == 0).sum().item())
        if n_iso > 0:
            print(f"  WARNING: {n_iso} isolated POIs (will get no graph signal)")

    return edge_index
