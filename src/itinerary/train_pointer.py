"""Training loop for the Strategy-B pointer itinerary model.

Teacher-forced sequence cross-entropy over whole trajectories (length>=3 sessions),
early-stopped on validation pairs-F1. Mirrors the plumbing of src.train.train_model
but for the pointer model and the itinerary metric. See itinerary_plan.md (Strategy B).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.itinerary.eval_itinerary import fmt_itinerary_metrics
from src.itinerary.pointer_model import PointerItineraryModel, evaluate_pointer
from src.itinerary.query import build_eval_queries
from src.itinerary.run_itinerary import estimate_assumed_dt_hours
from src.itinerary.seq_dataset import ItinerarySeqDataset, seq_collate_fn


def make_seq_loaders(
    processed_dir: str,
    device: torch.device,
    batch_size: int = 64,
    min_len: int = 3,
    num_workers: int = 2,
) -> Tuple[DataLoader, List, List, torch.Tensor, Dict[str, Any], np.ndarray, float]:
    """Build the pointer training loader + val/test eval queries + graph + meta.

    Args:
        processed_dir: dir with train/val/test parquet, edge_index.pt,
            poi_coords.npy, meta.json.
        device: device for edge_index.
        batch_size: training batch size.
        min_len: minimum session length kept (default 3).
        num_workers: DataLoader workers.

    Returns:
        ``(train_loader, val_queries, test_queries, edge_index, meta, poi_coords,
        assumed_dt_hours)``. ``poi_coords`` and ``assumed_dt_hours`` are only used
        by the context-aware (B-v2) decoder.
    """
    train_df = pd.read_parquet(os.path.join(processed_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(processed_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(processed_dir, "test.parquet"))
    edge_index = torch.load(os.path.join(processed_dir, "edge_index.pt")).to(device)
    poi_coords = np.load(os.path.join(processed_dir, "poi_coords.npy"))
    assumed_dt = estimate_assumed_dt_hours(train_df)
    with open(os.path.join(processed_dir, "meta.json")) as f:
        meta = json.load(f)

    train_ds = ItinerarySeqDataset(train_df, min_len=min_len)
    val_queries = build_eval_queries(val_df, fixed_end=True, min_len=min_len)
    test_queries = build_eval_queries(test_df, fixed_end=True, min_len=min_len)
    print(
        f"pointer train sessions={len(train_ds):,} | "
        f"val queries={len(val_queries):,} | test queries={len(test_queries):,}"
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=seq_collate_fn, num_workers=num_workers,
    )
    return train_loader, val_queries, test_queries, edge_index, meta, poi_coords, assumed_dt


def train_one_epoch_pointer(
    model: PointerItineraryModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    edge_index: torch.Tensor,
    device: torch.device,
    grad_clip: float = 5.0,
) -> float:
    """One teacher-forced training epoch; returns mean per-batch loss.

    Loss is cross-entropy over the pointer logits at every real (non-padding) step.

    Args:
        model: PointerItineraryModel in train mode (set internally).
        loader: training DataLoader (seq batches).
        optimizer: torch optimizer.
        edge_index: (2, E) long graph on ``device``.
        device: torch device.
        grad_clip: max L2 gradient norm.

    Returns:
        Mean per-batch loss.
    """
    model.train()
    total, nb = 0.0, 0
    pbar = tqdm(loader, desc="train-ptr", leave=False)
    for batch in pbar:
        poi_seq = batch["poi_seq"].to(device)
        lengths = batch["lengths"].to(device)
        user_ids = batch["user_ids"].to(device)
        delta_d = batch["delta_d"].to(device)
        delta_t = batch["delta_t"].to(device)

        logits, targets, mask = model(
            poi_seq, lengths, user_ids, edge_index, delta_d=delta_d, delta_t=delta_t
        )
        # cross-entropy over valid steps only
        loss = F.cross_entropy(logits[mask], targets[mask])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item(); nb += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total / max(nb, 1)


def train_pointer_model(
    name: str,
    project_root: str,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 8,
    min_len: int = 3,
    beam: int = 3,
    use_context: bool = False,
    d_p: int = 128,
    d_u: int = 64,
    d_h: int = 128,
    d_c: int = 32,
    dropout: float = 0.2,
    num_workers: int = 2,
) -> Tuple[PointerItineraryModel, Dict[str, float], List[Dict[str, Any]]]:
    """Train the pointer itinerary model with early stopping on val pairs-F1.

    Reads ``{project_root}/data/processed/{name}/``; writes checkpoints to
    ``{project_root}/checkpoints/{name}_pointer{suffix}/`` and results to
    ``{project_root}/results/{name}_pointer{suffix}_test.json``, where ``suffix``
    is ``"_ctx"`` when ``use_context`` (B-v2) else ``""`` (B-v1).

    Args:
        name: city name (e.g. ``"NYC"``).
        project_root: root with data/, checkpoints/, results/.
        device: torch device.
        epochs, batch_size, lr, weight_decay, patience: training schedule.
        min_len: minimum session length for training + eval (default 3).
        beam: beam width used for the final test decode.
        use_context: B-v2 — feed Δd/Δt context into the decoder.
        d_p, d_u, d_h, d_c, dropout: model dims.
        num_workers: DataLoader workers.

    Returns:
        ``(model, test_metrics, history)`` — model has the best (val pairs-F1) weights.
    """
    suffix = "_ctx" if use_context else ""
    tag = f"pointer{suffix}"
    print(f"\n{'='*60}\nStrategy-B {tag} training on {name} (use_context={use_context})\n{'='*60}")
    proc = os.path.join(project_root, "data/processed", name)
    train_loader, val_q, test_q, edge_index, meta, coords, adt = make_seq_loaders(
        proc, device=device, batch_size=batch_size, min_len=min_len, num_workers=num_workers,
    )
    model = PointerItineraryModel(
        n_pois=meta["n_pois"], n_users=meta["n_users"],
        d_p=d_p, d_u=d_u, d_h=d_h, d_c=d_c, use_context=use_context, dropout=dropout,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}"
          + (f" | assumed_dt_hours={adt:.3f}" if use_context else ""))

    ev = lambda q, dec, bw=beam: evaluate_pointer(
        model, q, edge_index, device, decoder=dec, beam=bw, min_len=min_len,
        poi_coords=coords, assumed_dt_hours=adt,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ckpt_dir = os.path.join(project_root, "checkpoints", f"{name}_{tag}")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    best_pf1, best_epoch, no_improve = -1.0, -1, 0
    history: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch_pointer(model, train_loader, opt, edge_index, device)
        val_m = ev(val_q, "greedy")
        history.append({"epoch": epoch, "train_loss": loss, **{f"val_{k}": v for k, v in val_m.items()}})
        print(f"[{name}] epoch {epoch:02d} | loss={loss:.4f} | val: {fmt_itinerary_metrics(val_m)}")

        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "epoch": epoch, "history": history},
                   os.path.join(ckpt_dir, "latest.pt"))
        if val_m["pairs-F1"] > best_pf1:
            best_pf1, best_epoch, no_improve = val_m["pairs-F1"], epoch, 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stop at epoch {epoch}. Best epoch {best_epoch} (val pairs-F1={best_pf1:.4f})")
            break

    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device))
    test_greedy = ev(test_q, "greedy")
    test_beam = ev(test_q, "beam")
    print(f"\n[{name}] TEST greedy (best ep {best_epoch}): {fmt_itinerary_metrics(test_greedy)}")
    print(f"[{name}] TEST beam{beam}: {fmt_itinerary_metrics(test_beam)}")

    pd.DataFrame(history).to_csv(os.path.join(results_dir, f"{name}_{tag}_history.csv"), index=False)
    with open(os.path.join(results_dir, f"{name}_{tag}_test.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "min_len": min_len, "use_context": use_context,
                   "greedy": test_greedy, f"beam{beam}": test_beam}, f, indent=2)

    return model, test_greedy, history
