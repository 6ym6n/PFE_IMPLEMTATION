"""Sequence dataset for Strategy-B pointer training.

Unlike the next-POI ``POISessionDataset`` (which makes L-1 prefix->next pairs),
the pointer model is trained on WHOLE trajectories: one example per session.
See itinerary_plan.md (Strategy B).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ItinerarySeqDataset(Dataset):
    """One example per session: the full ordered POI sequence + its user.

    Args:
        df: sessionized DataFrame with columns ``session_id``, ``timestamp``,
            ``user_idx``, ``poi_idx``.
        min_len: skip sessions shorter than this (default 3 — the pointer task
            needs a start, at least one intermediate, and an end to be non-trivial).
        max_seq_len: hard cap on sequence length (left-truncate longer sessions).
    """

    def __init__(self, df: pd.DataFrame, min_len: int = 3, max_seq_len: int = 100) -> None:
        self.examples: List[Dict[str, Any]] = []
        for _sid, g in df.groupby("session_id"):
            seq = g.sort_values("timestamp")["poi_idx"].astype(int).tolist()
            if len(seq) < min_len:
                continue
            if len(seq) > max_seq_len:
                seq = seq[-max_seq_len:]
            self.examples.append(
                {"user": int(g["user_idx"].iloc[0]), "poi_seq": np.asarray(seq, dtype=np.int64)}
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def seq_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Right-pad variable-length POI sequences to the batch max.

    Padding value for ``poi_seq`` is 0 (a valid POI row in the GCN matrix; the
    loss masks padded steps via ``lengths`` so the pad value is never scored).

    Args:
        batch: list of dicts from :class:`ItinerarySeqDataset`.

    Returns:
        dict with tensors:
            - ``poi_seq``  (B, T_max) long  — padded ordered sequences
            - ``lengths``  (B,)       long  — true session lengths (all ≥ min_len)
            - ``user_ids`` (B,)       long
    """
    lengths = torch.tensor([len(b["poi_seq"]) for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    B = len(batch)
    poi_seq = torch.zeros(B, max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["poi_seq"])
        poi_seq[i, :L] = torch.from_numpy(b["poi_seq"])
    return {
        "poi_seq": poi_seq,
        "lengths": lengths,
        "user_ids": torch.tensor([b["user"] for b in batch], dtype=torch.long),
    }
