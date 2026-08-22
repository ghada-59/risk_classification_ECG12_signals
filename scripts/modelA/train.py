"""Training entry point for Modèle A.

Usage
-----
    python -m scripts.modelA.train                       # full training
    python -m scripts.modelA.train --epochs 5            # quick sanity run
    python -m scripts.modelA.train --max-samples 2000    # subset for smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .config import CLASS_NAMES, MODELS_DIR, REPORTS_DIR, TRAINING
from .dataset import build_split_datasets, class_weights_from_df
from .model import ModelA, count_parameters


logger = logging.getLogger("train")


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            "weight", weight if weight is not None else torch.tensor([])
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = torch.log_softmax(logits, dim=-1)
        p = log_p.exp()
        targets_onehot = nn.functional.one_hot(targets, num_classes=logits.size(-1)).float()
        focal_term = (1.0 - p) ** self.gamma
        loss = -(focal_term * log_p * targets_onehot).sum(dim=-1)
        if self.weight.numel() > 0:
            w = self.weight.to(logits.device)[targets]
            loss = loss * w
        return loss.mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro"
        ))
    except ValueError:
        return float("nan")


@torch.no_grad()
def evaluate_loader(
    model: ModelA, loader: DataLoader, device: str
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss()
    y_true: list[int] = []
    y_prob: list[np.ndarray] = []
    for sig, feats, label in loader:
        sig = sig.to(device, non_blocking=True)
        feats = feats.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits = model(sig, feats)
        losses.append(criterion(logits, label).item())
        y_true.append(label.cpu().numpy())
        y_prob.append(torch.softmax(logits, dim=-1).cpu().numpy())
    y_true_arr = np.concatenate(y_true)
    y_prob_arr = np.concatenate(y_prob, axis=0)
    return (
        float(np.mean(losses)),
        _macro_auc(y_true_arr, y_prob_arr),
        y_true_arr,
        y_prob_arr,
    )


def train_one_epoch(
    model: ModelA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    grad_clip: float,
) -> float:
    model.train()
    losses: list[float] = []
    bar = tqdm(loader, desc="train", unit="batch", leave=False)
    for sig, feats, label in bar:
        sig = sig.to(device, non_blocking=True)
        feats = feats.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(sig, feats)
        loss = criterion(logits, label)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
        bar.set_postfix(loss=f"{loss.item():.3f}")
    return float(np.mean(losses)) if losses else float("nan")


def _subsample(dataset, max_samples: int):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    idx = list(range(len(dataset)))
    random.Random(TRAINING.seed).shuffle(idx)
    return Subset(dataset, idx[:max_samples])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Modèle A.")
    parser.add_argument("--epochs", type=int, default=TRAINING.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    parser.add_argument("--lr", type=float, default=TRAINING.lr)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap on samples per split (smoke test).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=TRAINING.num_workers)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    set_seed(TRAINING.seed)
    device = args.device

    train_ds, val_ds, test_ds, full_df = build_split_datasets()

    train_ds_sub = _subsample(train_ds, args.max_samples)
    val_ds_sub = _subsample(val_ds, args.max_samples)
    test_ds_sub = _subsample(test_ds, args.max_samples)

    weights = class_weights_from_df(
        full_df[full_df["strat_fold"].isin(TRAINING.train_folds)]
    )
    logger.info("Class weights: %s", weights.tolist())

    train_loader = DataLoader(
        train_ds_sub, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds_sub, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device != "cpu"),
    )
    test_loader = DataLoader(
        test_ds_sub, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device != "cpu"),
    )

    model = ModelA().to(device)
    logger.info("Modèle A — params=%s", f"{count_parameters(model):,}")

    criterion = FocalLoss(gamma=TRAINING.focal_gamma, weight=weights).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=TRAINING.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS_DIR / "training_log.csv"
    ckpt_path = MODELS_DIR / "modelA_best.pt"
    log_path.write_text(
        "epoch,train_loss,val_loss,val_auc,lr,elapsed_s\n", encoding="utf-8"
    )

    best_auc = -float("inf")
    epochs_since_best = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, TRAINING.grad_clip
        )
        val_loss, val_auc, _, _ = evaluate_loader(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            "epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f  lr=%.2e  (%.1fs)",
            epoch, args.epochs, train_loss, val_loss, val_auc, lr_now, elapsed,
        )
        with log_path.open("a", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, val_loss, val_auc, lr_now, elapsed]
            )

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_since_best = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "val_loss": val_loss,
                    "config": {
                        "model": vars(model.cfg),
                        "training": vars(TRAINING),
                    },
                },
                ckpt_path,
            )
            logger.info("  saved best to %s (val_auc=%.4f)", ckpt_path, val_auc)
        else:
            epochs_since_best += 1
            if epochs_since_best >= TRAINING.early_stop_patience:
                logger.info("Early stopping after %d epochs without improvement.",
                            epochs_since_best)
                break

    logger.info("Loading best checkpoint for test eval...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_auc, y_true, y_prob = evaluate_loader(model, test_loader, device)
    logger.info("TEST  loss=%.4f  macro_auc=%.4f", test_loss, test_auc)

    summary = {
        "best_val_auc": best_auc,
        "test_loss": test_loss,
        "test_macro_auc": test_auc,
        "n_params": count_parameters(model),
        "class_names": list(CLASS_NAMES),
        "checkpoint": str(ckpt_path),
        "epochs_trained": ckpt["epoch"],
    }
    (REPORTS_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        REPORTS_DIR / "test_predictions.npz",
        y_true=y_true,
        y_prob=y_prob,
    )
    logger.info("Done. Summary saved to %s", REPORTS_DIR / "training_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
