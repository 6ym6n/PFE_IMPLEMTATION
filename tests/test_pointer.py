"""Tests for the Strategy-B pointer itinerary model.

Covers the sequence dataset, the teacher-forced forward (shapes + backward),
decode invariants (loop-free, length, fixed-end-last, greedy==beam1), and a
tiny overfit-sanity check that training loss decreases on a memorizable set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.itinerary.pointer_model import (
    PointerItineraryModel,
    evaluate_pointer,
    pointer_rollout_beam,
    pointer_rollout_greedy,
)
from src.itinerary.query import ItineraryQuery
from src.itinerary.seq_dataset import ItinerarySeqDataset, seq_collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# sequence dataset
# --------------------------------------------------------------------------
def _sessions_df():
    rows = []
    specs = [(1, 0, [0, 1, 2, 3]), (2, 1, [4, 5, 6]), (3, 1, [7, 8])]  # last is len-2
    for sid, u, seq in specs:
        for i, p in enumerate(seq):
            rows.append({"session_id": sid, "user_idx": u, "poi_idx": p,
                         "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=sid, minutes=10 * i)})
    return pd.DataFrame(rows)


def test_seq_dataset_keeps_len_ge_3_only():
    ds = ItinerarySeqDataset(_sessions_df(), min_len=3)
    assert len(ds) == 2                       # len-2 session dropped
    assert list(ds[0]["poi_seq"]) == [0, 1, 2, 3]
    assert ds[0]["user"] == 0


def test_seq_collate_pads_and_lengths():
    ds = ItinerarySeqDataset(_sessions_df(), min_len=3)
    batch = seq_collate_fn([ds[0], ds[1]])    # lengths 4 and 3
    assert batch["poi_seq"].shape == (2, 4)
    assert batch["lengths"].tolist() == [4, 3]
    assert batch["poi_seq"][1, 3].item() == 0  # padding
    assert batch["poi_seq"].dtype == torch.long


# --------------------------------------------------------------------------
# model fixture + forward
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    n_pois, n_users = 30, 4
    model = PointerItineraryModel(n_pois, n_users, d_p=32, d_u=16, d_h=32).to(DEVICE).eval()
    src = list(range(n_pois)) + list(range(n_pois - 1))
    dst = list(range(n_pois)) + list(range(1, n_pois))
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long, device=DEVICE)
    return model, edge_index, n_pois


def test_forward_shapes_and_backward(tiny):
    model, edge_index, n_pois = tiny
    model.train()
    poi_seq = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 0]], device=DEVICE)  # 2nd is len-3 padded
    lengths = torch.tensor([4, 3], device=DEVICE)
    users = torch.tensor([0, 1], device=DEVICE)
    logits, targets, mask = model(poi_seq, lengths, users, edge_index)
    assert logits.shape == (2, 3, n_pois)
    assert targets.shape == (2, 3)
    assert mask.tolist() == [[True, True, True], [True, True, False]]  # padded step masked
    loss = torch.nn.functional.cross_entropy(logits[mask], targets[mask])
    loss.backward()
    assert model.gcn.embedding.weight.grad is not None
    assert model.user_emb.weight.grad is not None
    assert model.out_proj.weight.grad is not None
    assert model.init_proj.weight.grad is not None
    model.eval()


def _q(start, end, K, gt=None):
    return ItineraryQuery(user_idx=0, start_poi=start, end_poi=end, K=K,
                          ground_truth=gt or list(range(K)))


def test_decode_invariants(tiny):
    model, edge_index, _ = tiny
    for start, end, K in [(3, 7, 5), (0, 1, 4), (10, 2, 6)]:
        q = _q(start, end, K)
        for r in (pointer_rollout_greedy(model, q, edge_index, DEVICE),
                  pointer_rollout_beam(model, q, edge_index, DEVICE, beam=4)):
            assert len(r) == K
            assert len(set(r)) == len(r)        # loop-free
            assert r[0] == start and r[-1] == end
            assert end not in r[:-1]             # end reserved for last position only


def test_greedy_equals_beam1(tiny):
    model, edge_index, _ = tiny
    q = _q(2, 9, 5)
    assert (pointer_rollout_greedy(model, q, edge_index, DEVICE)
            == pointer_rollout_beam(model, q, edge_index, DEVICE, beam=1))


def test_degenerate_k1(tiny):
    model, edge_index, _ = tiny
    q = _q(5, 5, 1, gt=[5])
    assert pointer_rollout_greedy(model, q, edge_index, DEVICE) == [5]
    assert pointer_rollout_beam(model, q, edge_index, DEVICE, beam=3) == [5]


def test_evaluate_pointer_keys(tiny):
    model, edge_index, _ = tiny
    qs = [_q(3, 7, 5), _q(2, 9, 4, gt=[2, 0, 1, 9])]
    m = evaluate_pointer(model, qs, edge_index, DEVICE, decoder="greedy")
    for k in ("pairs-F1", "set-F1", "exact-match", "feasibility", "n"):
        assert k in m
    assert m["n"] == 2.0 and m["feasibility"] == 1.0


def test_overfit_tiny_set_loss_decreases():
    # The model should be able to drive training loss down on a small memorizable set.
    torch.manual_seed(0)
    n_pois, n_users = 12, 2
    model = PointerItineraryModel(n_pois, n_users, d_p=32, d_u=16, d_h=32).to(DEVICE)
    ei = torch.tensor([[i for i in range(n_pois)] + [i for i in range(n_pois - 1)],
                       [i for i in range(n_pois)] + [i + 1 for i in range(n_pois - 1)]],
                      dtype=torch.long, device=DEVICE)
    # symmetrize
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    seqs = [[0, 3, 5, 9], [1, 4, 6, 8], [2, 7, 10, 11]]
    poi_seq = torch.tensor(seqs, device=DEVICE)
    lengths = torch.tensor([4, 4, 4], device=DEVICE)
    users = torch.tensor([0, 1, 0], device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    losses = []
    for _ in range(40):
        logits, targets, mask = model(poi_seq, lengths, users, ei)
        loss = torch.nn.functional.cross_entropy(logits[mask], targets[mask])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.6, f"loss did not drop enough: {losses[0]:.3f} -> {losses[-1]:.3f}"
