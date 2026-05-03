"""Submodules for the next-POI architecture.

Mirrors Section 6 of implementation_guide.md exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class POIGraphEncoder(nn.Module):
    """2-layer GCN over the POI graph.

    Holds a learnable POI embedding table E ∈ R^{|V| x d_p} and propagates it
    through ``n_layers`` GCN layers using the hybrid adjacency. Forward returns
    POI features for the entire vocabulary in a single pass — call once per
    minibatch and gather rows by ``poi_idx``.

    Args:
        n_pois: vocabulary size |V|.
        d_p: POI embedding / hidden dimension.
        n_layers: number of GCN layers (default 2 per spec).
        dropout: dropout probability between layers.
    """

    def __init__(self, n_pois: int, d_p: int, n_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_pois, d_p)
        self.convs = nn.ModuleList([GCNConv(d_p, d_p) for _ in range(n_layers)])
        self.dropout = dropout
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Propagate POI embeddings through the GCN.

        Args:
            edge_index: (2, E) long tensor of directed edges (PyG convention).

        Returns:
            (n_pois, d_p) tensor of POI features after ``n_layers`` GCN passes.
        """
        x = self.embedding.weight  # (n_pois, d_p)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class ContextEncoder(nn.Module):
    """Lift Δd and Δt scalars into a d_c-dim context embedding per step.

    Two parallel MLPs each map a scalar to ``d_c // 2`` dims; the two halves
    are concatenated. For session boundaries (first POI), the input scalars
    are 0 by convention (handled upstream in preprocessing/dataset).

    Args:
        d_c: total context dim. Must be even (split into two halves).
    """

    def __init__(self, d_c: int = 32) -> None:
        super().__init__()
        d_half = d_c // 2
        self.mlp_d = nn.Sequential(
            nn.Linear(1, d_half), nn.ReLU(),
            nn.Linear(d_half, d_half),
        )
        self.mlp_t = nn.Sequential(
            nn.Linear(1, d_half), nn.ReLU(),
            nn.Linear(d_half, d_half),
        )

    def forward(self, delta_d: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """Encode per-step spatial and temporal gaps.

        Args:
            delta_d: (B, T) float tensor — haversine distance from previous POI (km).
            delta_t: (B, T) float tensor — time gap from previous POI (hours).

        Returns:
            (B, T, d_c) tensor of per-step context embeddings.
        """
        d = self.mlp_d(delta_d.unsqueeze(-1))  # (B, T, d_half)
        t = self.mlp_t(delta_t.unsqueeze(-1))  # (B, T, d_half)
        return torch.cat([d, t], dim=-1)        # (B, T, d_c)
