"""Eval-metric correctness on a hand-crafted toy ranking."""

from __future__ import annotations

import math

import torch

from src.eval import evaluate_model


class _ConstantModel(torch.nn.Module):
    """Returns a fixed (B, n_pois) logits tensor regardless of input.

    Useful for checking that HR / NDCG / MRR are computed correctly given
    known ranks.
    """

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self._logits = logits

    def forward(self, poi_ids, delta_d, delta_t, user_ids, lengths, edge_index):
        B = poi_ids.shape[0]
        return self._logits[:B]


def test_evaluate_model_metrics_on_known_ranks() -> None:
    # 3 examples, 5 POIs. We control the rank of each target by choosing logits.
    # Example 0: target 0, logits [10, 5, 1, 0, -1]   → target rank 1
    # Example 1: target 2, logits [1, 1, 1, 0.5, 2]   → target rank 3 (one strict >, plus ties don't count strictly higher; 2 strictly higher than 1 → rank 3? Recompute.)
    # Use unambiguous ranks:
    logits = torch.tensor([
        [10.0, 5.0, 1.0, 0.0, -1.0],   # target 0 → rank 1
        [0.0, 1.0, 5.0, 3.0, 4.0],     # target 2 → rank 1 (5 is highest)
        [0.0, 1.0, 2.0, 3.0, 4.0],     # target 0 → rank 5
    ])
    targets = torch.tensor([0, 2, 0])

    # Build a minimal loader
    class _DS(torch.utils.data.Dataset):
        def __len__(self): return 3
        def __getitem__(self, i):
            return {
                "poi_ids": torch.zeros(1, dtype=torch.long),
                "delta_d": torch.zeros(1),
                "delta_t": torch.zeros(1),
                "user": 0,
                "target": int(targets[i].item()),
            }

    def collate(batch):
        return {
            "poi_ids": torch.stack([b["poi_ids"] for b in batch]),
            "delta_d": torch.stack([b["delta_d"] for b in batch]),
            "delta_t": torch.stack([b["delta_t"] for b in batch]),
            "user_ids": torch.tensor([b["user"] for b in batch], dtype=torch.long),
            "lengths": torch.tensor([1] * len(batch), dtype=torch.long),
            "targets": torch.tensor([b["target"] for b in batch], dtype=torch.long),
        }

    loader = torch.utils.data.DataLoader(_DS(), batch_size=3, collate_fn=collate)
    model = _ConstantModel(logits)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    m = evaluate_model(model, loader, edge_index, device=torch.device("cpu"),
                       ks=(1, 5))

    # Ranks: [1, 1, 5]
    # HR@1 = 2/3 (first two), HR@5 = 3/3 (all)
    assert math.isclose(m["HR@1"], 2 / 3, abs_tol=1e-6)
    assert math.isclose(m["HR@5"], 1.0, abs_tol=1e-6)
    # MRR = (1/1 + 1/1 + 1/5) / 3
    assert math.isclose(m["MRR"], (1 + 1 + 0.2) / 3, abs_tol=1e-6)
    # NDCG@5 for rank 1: 1/log2(2) = 1.0; rank 5: 1/log2(6)
    expected_ndcg5 = (1.0 + 1.0 + 1.0 / math.log2(6)) / 3
    assert math.isclose(m["NDCG@5"], expected_ndcg5, abs_tol=1e-6)
