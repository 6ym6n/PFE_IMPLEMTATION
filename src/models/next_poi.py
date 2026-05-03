"""Full next-POI architecture: GCN + context + user embedding + GRU + MLP head.

Mirrors Section 6 of implementation_guide.md exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.components import ContextEncoder, POIGraphEncoder


class NextPOIModel(nn.Module):
    """TLMR-faithful next-POI recommender.

    Forward pipeline (all steps in a single forward pass):
        1. GCN over the full POI graph → POI feature matrix (|V|, d_p).
        2. Per step t: gather GCN row for p_t, build context c_t from Δd, Δt,
           concat → x_t ∈ R^(d_p + d_c).
        3. Run a GRU over the (padded, packed) sequence → final hidden state h_T.
        4. Concatenate user embedding e_u → z = [h_T ; e_u] ∈ R^(d_h + d_u).
        5. MLP head → logits over |V| POIs.

    Args:
        n_pois: vocabulary size |V|.
        n_users: number of users |U|.
        d_p: POI embedding dim (default 128).
        d_u: user embedding dim (default 64).
        d_c: context dim (default 32).
        d_h: GRU hidden dim (default 128).
        d_hidden: MLP head hidden dim (default 256).
        dropout: dropout rate (default 0.2).
    """

    def __init__(
        self,
        n_pois: int,
        n_users: int,
        d_p: int = 128,
        d_u: int = 64,
        d_c: int = 32,
        d_h: int = 128,
        d_hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gcn = POIGraphEncoder(n_pois, d_p, n_layers=2, dropout=dropout)
        self.context = ContextEncoder(d_c)
        self.user_emb = nn.Embedding(n_users, d_u)
        self.gru = nn.GRU(d_p + d_c, d_h, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_h + d_u, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_pois),
        )
        nn.init.xavier_uniform_(self.user_emb.weight)

    def forward(
        self,
        poi_ids: torch.Tensor,
        delta_d: torch.Tensor,
        delta_t: torch.Tensor,
        user_ids: torch.Tensor,
        lengths: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Run a full forward pass.

        Args:
            poi_ids:    (B, T) long  — sequence of POI indices, padded with 0.
            delta_d:    (B, T) float — spatial gaps (km), padded with 0.
            delta_t:    (B, T) float — temporal gaps (h), padded with 0.
            user_ids:   (B,)   long  — user index per sequence.
            lengths:    (B,)   long  — actual (unpadded) sequence lengths, all ≥ 1.
            edge_index: (2, E) long  — full hybrid POI graph (PyG format).

        Returns:
            (B, n_pois) float tensor of unnormalized logits.
        """
        poi_features = self.gcn(edge_index)            # (V, d_p)
        seq = poi_features[poi_ids]                    # (B, T, d_p)
        ctx = self.context(delta_d, delta_t)           # (B, T, d_c)
        x = torch.cat([seq, ctx], dim=-1)              # (B, T, d_p + d_c)

        # Pack to handle variable lengths. lengths must live on CPU for PyTorch.
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, h_T = self.gru(packed)                      # h_T: (1, B, d_h)
        h_T = h_T.squeeze(0)                           # (B, d_h)

        u = self.user_emb(user_ids)                    # (B, d_u)
        z = torch.cat([h_T, u], dim=-1)                # (B, d_h + d_u)
        return self.head(z)                            # (B, V)
