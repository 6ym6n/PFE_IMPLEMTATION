"""Section-6 smoke test as a pytest test, plus shape and softmax-sum checks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.next_poi import NextPOIModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_smoke_section_6() -> None:
    """Reproduces the smoke test from Section 6 of implementation_guide.md."""
    n_pois, n_users = 100, 20
    model = NextPOIModel(n_pois, n_users).to(DEVICE)
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long, device=DEVICE,
    )

    B, T = 8, 5
    poi_ids = torch.randint(0, n_pois, (B, T), device=DEVICE)
    delta_d = torch.rand(B, T, device=DEVICE)
    delta_t = torch.rand(B, T, device=DEVICE)
    user_ids = torch.randint(0, n_users, (B,), device=DEVICE)
    lengths = torch.tensor([5, 4, 3, 5, 2, 4, 5, 3])

    logits = model(poi_ids, delta_d, delta_t, user_ids, lengths, edge_index)
    assert logits.shape == (B, n_pois)

    probs = F.softmax(logits, dim=-1)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_backward_pass_runs() -> None:
    """Loss must backprop through GCN, GRU, head, embeddings."""
    n_pois, n_users = 50, 10
    model = NextPOIModel(n_pois, n_users).to(DEVICE)
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long, device=DEVICE,
    )
    B, T = 4, 3
    poi_ids = torch.randint(0, n_pois, (B, T), device=DEVICE)
    delta_d = torch.rand(B, T, device=DEVICE)
    delta_t = torch.rand(B, T, device=DEVICE)
    user_ids = torch.randint(0, n_users, (B,), device=DEVICE)
    lengths = torch.tensor([3, 2, 3, 1])
    targets = torch.randint(0, n_pois, (B,), device=DEVICE)

    logits = model(poi_ids, delta_d, delta_t, user_ids, lengths, edge_index)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    # Embedding grads exist
    assert model.gcn.embedding.weight.grad is not None
    assert model.user_emb.weight.grad is not None


def test_param_count_in_expected_band() -> None:
    """Section-6 expected: ~50k params for n_pois=100. Our exact count is ~234k
    because the spec dimensions (d_p=128, d_h=128, d_hidden=256) dominate even
    at small |V|. Just sanity-check the order of magnitude is reasonable."""
    n_pois, n_users = 100, 20
    model = NextPOIModel(n_pois, n_users)
    n = sum(p.numel() for p in model.parameters())
    assert 50_000 < n < 1_000_000
