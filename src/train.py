"""Training loop for the next-POI baseline.

Mirrors Section 8 of implementation_guide.md exactly.

The high-level entry point is :func:`train_model`, which loads processed data
from a city directory, instantiates the model, trains with early stopping on
val HR@10, and writes checkpoints + metric history to disk.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import POISessionDataset, collate_fn
from src.eval import evaluate_model, fmt_metrics
from src.models.next_poi import NextPOIModel


def make_loaders(
    processed_dir: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 2,
    max_seq_len: int = 100,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, Dict[str, Any]]:
    """Load preprocessed parquets + edge_index + meta for one city.

    Args:
        processed_dir: directory containing
            ``train.parquet``, ``val.parquet``, ``test.parquet``,
            ``edge_index.pt``, ``meta.json``.
        device: target device for ``edge_index`` (kept on device, fed each batch).
        batch_size: batch size for all three loaders.
        num_workers: DataLoader worker count.
        max_seq_len: max history length for :class:`POISessionDataset`.

    Returns:
        ``(train_loader, val_loader, test_loader, edge_index, meta)``.
    """
    train_df = pd.read_parquet(os.path.join(processed_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(processed_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(processed_dir, "test.parquet"))
    edge_index = torch.load(os.path.join(processed_dir, "edge_index.pt")).to(device)
    with open(os.path.join(processed_dir, "meta.json")) as f:
        meta = json.load(f)

    train_ds = POISessionDataset(train_df, max_seq_len=max_seq_len)
    val_ds = POISessionDataset(val_df, max_seq_len=max_seq_len)
    test_ds = POISessionDataset(test_df, max_seq_len=max_seq_len)

    print(
        f"train={len(train_ds):,} | val={len(val_ds):,} | test={len(test_ds):,} examples"
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, edge_index, meta


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    edge_index: torch.Tensor,
    device: torch.device,
    grad_clip: float = 5.0,
) -> float:
    """Run one training epoch; return mean cross-entropy loss.

    Applies gradient clipping at ``grad_clip`` (L2 norm). Loss is averaged over
    minibatches (NOT examples) to match the guide.

    Args:
        model: model in training mode (set internally).
        loader: training DataLoader.
        optimizer: torch optimizer.
        edge_index: (2, E) long tensor on ``device``.
        device: torch device for batch tensors.
        grad_clip: max L2 gradient norm.

    Returns:
        Mean per-batch loss as a Python float.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        poi_ids = batch["poi_ids"].to(device)
        delta_d = batch["delta_d"].to(device)
        delta_t = batch["delta_t"].to(device)
        user_ids = batch["user_ids"].to(device)
        lengths = batch["lengths"]
        targets = batch["targets"].to(device)

        logits = model(poi_ids, delta_d, delta_t, user_ids, lengths, edge_index)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / max(n_batches, 1)


def train_model(
    name: str,
    project_root: str,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 8,
    d_p: int = 128,
    d_u: int = 64,
    d_c: int = 32,
    d_h: int = 128,
    d_hidden: int = 256,
    dropout: float = 0.2,
    num_workers: int = 2,
) -> Tuple[torch.nn.Module, Dict[str, float], List[Dict[str, Any]]]:
    """Train the next-POI model on one city with early stopping on val HR@10.

    Reads from ``{project_root}/data/processed/{name}/`` and writes to
    ``{project_root}/checkpoints/{name}/`` and ``{project_root}/results/``.

    Args:
        name: city name (e.g. ``"NYC"``, ``"TKY"``).
        project_root: root directory with ``data/``, ``checkpoints/``, ``results/``.
        device: torch device.
        epochs, batch_size, lr, weight_decay, patience: training schedule.
        d_p, d_u, d_c, d_h, d_hidden, dropout: model hyperparameters.
        num_workers: DataLoader workers.

    Returns:
        ``(model, test_metrics, history)``:
            - model loaded with the best (by val HR@10) checkpoint.
            - test_metrics: dict of HR@k, NDCG@k, MRR on the test set.
            - history: list of per-epoch dicts ``{epoch, train_loss, **val_metrics}``.
    """
    print(f"\n{'=' * 60}\nTraining on {name}\n{'=' * 60}")
    processed_dir = os.path.join(project_root, "data/processed", name)
    train_loader, val_loader, test_loader, edge_index, meta = make_loaders(
        processed_dir, device=device, batch_size=batch_size, num_workers=num_workers,
    )

    model = NextPOIModel(
        n_pois=meta["n_pois"],
        n_users=meta["n_users"],
        d_p=d_p, d_u=d_u, d_c=d_c, d_h=d_h, d_hidden=d_hidden, dropout=dropout,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ckpt_dir = os.path.join(project_root, "checkpoints", name)
    results_dir = os.path.join(project_root, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    best_hr10 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, edge_index, device)
        val_metrics = evaluate_model(model, val_loader, edge_index, device)
        history.append({"epoch": epoch, "train_loss": loss, **val_metrics})
        print(
            f"[{name}] Epoch {epoch:02d} | train_loss={loss:.4f} "
            f"| val: {fmt_metrics(val_metrics)}"
        )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
            },
            os.path.join(ckpt_dir, "latest.pt"),
        )

        if val_metrics["HR@10"] > best_hr10:
            best_hr10 = val_metrics["HR@10"]
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch} (val HR@10={best_hr10:.4f})"
            )
            break

    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt")))
    test_metrics = evaluate_model(model, test_loader, edge_index, device)
    print(f"\n[{name}] TEST (best epoch {best_epoch}): {fmt_metrics(test_metrics)}")

    pd.DataFrame(history).to_csv(
        os.path.join(results_dir, f"{name}_history.csv"), index=False,
    )
    with open(os.path.join(results_dir, f"{name}_test.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, **test_metrics}, f, indent=2)

    return model, test_metrics, history
