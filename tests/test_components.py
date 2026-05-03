"""Shape tests for POIGraphEncoder and ContextEncoder."""

from __future__ import annotations

import torch

from src.models.components import ContextEncoder, POIGraphEncoder


def test_poi_graph_encoder_output_shape() -> None:
    n_pois, d_p = 100, 128
    enc = POIGraphEncoder(n_pois=n_pois, d_p=d_p, n_layers=2, dropout=0.2)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 99], [1, 0, 3, 2, 99, 0]], dtype=torch.long,
    )
    out = enc(edge_index)
    assert out.shape == (n_pois, d_p)
    assert out.dtype == torch.float32


def test_poi_graph_encoder_n_layers() -> None:
    enc = POIGraphEncoder(n_pois=10, d_p=8, n_layers=2)
    assert len(enc.convs) == 2


def test_context_encoder_output_shape() -> None:
    d_c = 32
    enc = ContextEncoder(d_c=d_c)
    B, T = 4, 7
    delta_d = torch.rand(B, T)
    delta_t = torch.rand(B, T)
    out = enc(delta_d, delta_t)
    assert out.shape == (B, T, d_c)


def test_context_encoder_zero_input_runs() -> None:
    """Session boundaries pass Δd=Δt=0 — must not NaN."""
    enc = ContextEncoder(d_c=32)
    out = enc(torch.zeros(2, 3), torch.zeros(2, 3))
    assert torch.isfinite(out).all()
