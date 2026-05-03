"""Session → (history, target) examples + padding collate function.

Mirrors Section 7 of implementation_guide.md exactly.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class POISessionDataset(Dataset):
    """Materialize next-POI training examples from sessionized check-ins.

    Each session of length L yields L-1 examples: prefix of length 1..L-1 → next.
    All examples reference the SAME ``user_idx`` (sessions belong to one user).
    Long histories are truncated on the left to ``max_seq_len``.

    Args:
        df: sessionized DataFrame with columns ``session_id``, ``timestamp``,
            ``user_idx``, ``poi_idx``, ``delta_d``, ``delta_t``.
        max_seq_len: maximum kept history length (default 100).
    """

    def __init__(self, df: pd.DataFrame, max_seq_len: int = 100) -> None:
        self.examples: List[Dict[str, Any]] = []
        for _, g in df.groupby("session_id"):
            g = g.sort_values("timestamp")
            poi = g["poi_idx"].values
            dd = g["delta_d"].values.astype(np.float32)
            dt = g["delta_t"].values.astype(np.float32)
            user = int(g["user_idx"].iloc[0])

            for i in range(1, len(poi)):
                start = max(0, i - max_seq_len)
                self.examples.append(
                    {
                        "user": user,
                        "history_pois": poi[start:i],
                        "history_dd": dd[start:i],
                        "history_dt": dt[start:i],
                        "target": int(poi[i]),
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Right-pad variable-length histories to the batch max length.

    Pad value for ``poi_ids`` is 0 (any valid POI index would work since the
    GRU consumes only ``lengths`` real steps; we still index to a valid row of
    the GCN feature matrix). Δd, Δt pad with 0.

    Args:
        batch: list of dicts as produced by :class:`POISessionDataset`.

    Returns:
        dict with tensors:
            - ``poi_ids``  (B, T_max) long
            - ``delta_d``  (B, T_max) float
            - ``delta_t``  (B, T_max) float
            - ``user_ids`` (B,)       long
            - ``lengths``  (B,)       long  — real history lengths, all ≥ 1
            - ``targets``  (B,)       long  — next POI index
    """
    lengths = torch.tensor([len(b["history_pois"]) for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    B = len(batch)

    poi_ids = torch.zeros(B, max_len, dtype=torch.long)
    delta_d = torch.zeros(B, max_len, dtype=torch.float)
    delta_t = torch.zeros(B, max_len, dtype=torch.float)

    for i, b in enumerate(batch):
        L = len(b["history_pois"])
        poi_ids[i, :L] = torch.from_numpy(b["history_pois"]).long()
        delta_d[i, :L] = torch.from_numpy(b["history_dd"])
        delta_t[i, :L] = torch.from_numpy(b["history_dt"])

    return {
        "poi_ids": poi_ids,
        "delta_d": delta_d,
        "delta_t": delta_t,
        "user_ids": torch.tensor([b["user"] for b in batch], dtype=torch.long),
        "lengths": lengths,
        "targets": torch.tensor([b["target"] for b in batch], dtype=torch.long),
    }
