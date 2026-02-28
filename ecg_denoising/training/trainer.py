"""
training/trainer.py — Rotating Epoch Training Engine.

Each epoch trains on exactly one noise type, rotating through:
    clean → low → mid → high → clean → low → mid → high → ...

This continues until TOTAL_EPOCHS is exhausted.
No phases, no stages, no cycles — just a flat rotating sequence.

Why this prevents catastrophic forgetting:
  • The model sees clean→clean every 4th epoch, keeping identity mapping alive.
  • Low/mid/high alternate frequently so no single noise level dominates.
  • A single LR schedule runs across all epochs — no abrupt resets.
"""

import os, sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import config
from training.losses     import CombinedLoss
from training.metrics    import compute_all
from training.checkpoint import save_checkpoint, load_checkpoint
from training.logger     import load_state, append_epoch
from training.plotter    import update_plots


# ─────────────────────────────────────────────────────────────────────────────
# Epoch type display labels
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_LABEL = {
    "clean": "Clean → Clean  (identity)",
    "low":   f"Low noise     ({config.SNR_LOW} dB)",
    "mid":   f"Mid noise     ({config.SNR_MEDIUM} dB)",
    "high":  f"High noise    ({config.SNR_HIGH} dB)",
}

_TYPE_COLOR = {
    "clean": "green",
    "low":   "cyan",
    "mid":   "yellow",
    "high":  "red",
}


# ─────────────────────────────────────────────────────────────────────────────
# Single-epoch pass
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer, device,
               train: bool, epoch: int, epoch_type: str):
    model.train() if train else model.eval()
    split = "Train" if train else "Val"

    total_loss = 0.0
    n_seen = 0
    all_pred, all_tgt, all_noisy = [], [], []
    is_noisy = (epoch_type != "clean")

    bar = tqdm(
        loader,
        desc=f"  Ep {epoch:03d}/{config.TOTAL_EPOCHS} [{split}][{epoch_type}]",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        colour=_TYPE_COLOR[epoch_type],
    )

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            # Handle embedding model that returns (output, commit_loss)
            if isinstance(pred, tuple):
                pred, commit_loss = pred
                loss = criterion(pred, y) + commit_loss
            else:
                loss = criterion(pred, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()

            b   = len(x)
            n_seen     += b
            total_loss += loss.item() * b
            all_pred.append(pred.detach())
            all_tgt.append(y.detach())
            if is_noisy:
                all_noisy.append(x.detach())

            bar.set_postfix(loss=f"{total_loss/n_seen:.5f}")

    bar.close()

    pred_cat  = torch.cat(all_pred, dim=0)
    tgt_cat   = torch.cat(all_tgt,  dim=0)
    noisy_cat = torch.cat(all_noisy, dim=0) if is_noisy else None

    metrics         = compute_all(pred_cat, tgt_cat, noisy_cat)
    metrics["loss"] = total_loss / max(n_seen, 1)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_training(model, all_loaders: dict, device):
    """
    Run rotating epoch training.

    Args:
        model       : nn.Module
        all_loaders : output of get_all_loaders() — keys: clean/low/mid/high,
                      each with "train"/"val" DataLoaders.
        device      : torch.device
    """
    state         = load_state()
    start_epoch   = state.get("current_epoch", 0) + 1
    best_val_loss = state.get("best_val_loss", float("inf"))
    no_improve    = state.get("no_improve", 0)

    # Clamp so a fully completed run can't start with an empty range
    start_epoch = min(start_epoch, config.TOTAL_EPOCHS)

    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode      = "min",
        patience  = config.SCHEDULER_PATIENCE,
        factor    = config.SCHEDULER_FACTOR,
        min_lr    = config.SCHEDULER_MIN_LR,
    )

    load_checkpoint(model, optimizer, device=device)

    rotation = config.EPOCH_ROTATION   # ["clean", "low", "mid", "high"]
    n_types  = len(rotation)

    print(f"\n{'═'*68}")
    print(f"  ROTATING EPOCH TRAINING")
    print(f"  Rotation : {' → '.join(rotation)}")
    print(f"  Epochs   : {start_epoch} → {config.TOTAL_EPOCHS}")
    print(f"  Device   : {device}")
    print(f"{'═'*68}\n")

    epoch_bar = tqdm(
        range(start_epoch, config.TOTAL_EPOCHS + 1),
        desc  = "  Epochs",
        unit  = "ep",
        leave = True,
        dynamic_ncols = True,
        colour = "green",
    )

    criterion = CombinedLoss()

    for epoch in epoch_bar:
        # Which type is this epoch?
        epoch_type = rotation[(epoch - 1) % n_types]
        train_loader = all_loaders[epoch_type]["train"]
        val_loader   = all_loaders[epoch_type]["val"]

        tqdm.write(
            f"\n  ── Epoch {epoch:03d}/{config.TOTAL_EPOCHS}  "
            f"type={epoch_type.upper():<5s}  "
            f"{_TYPE_LABEL[epoch_type]}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        tr = _run_epoch(model, train_loader, criterion, optimizer, device,
                        train=True,  epoch=epoch, epoch_type=epoch_type)
        vl = _run_epoch(model, val_loader,   criterion, optimizer, device,
                        train=False, epoch=epoch, epoch_type=epoch_type)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(vl["loss"])

        is_best = vl["loss"] < best_val_loss
        if is_best:
            best_val_loss = vl["loss"]
            no_improve    = 0
        else:
            no_improve += 1

        save_checkpoint(model, optimizer, epoch=epoch,
                        phase=epoch_type, loss=vl["loss"], is_best=is_best)

        state = append_epoch(
            state,
            epoch        = epoch,
            epoch_type   = epoch_type,
            train_metrics = tr,
            val_metrics   = vl,
            lr           = current_lr,
        )
        state["best_val_loss"] = best_val_loss
        state["no_improve"]    = no_improve

        update_plots(state["history"])

        star = " ★" if is_best else ""
        epoch_bar.set_postfix_str(
            f"type={epoch_type:<5s} | "
            f"T[loss={tr['loss']:.5f} prd={tr['prd']:.2f}% ρ={tr['pearson']:.4f}] | "
            f"V[loss={vl['loss']:.5f} snri={vl['snr_improve']:.2f}dB "
            f"ρ={vl['pearson']:.4f}]{star}"
        )

        if config.EARLY_STOP_PATIENCE and no_improve >= config.EARLY_STOP_PATIENCE:
            tqdm.write(
                f"\n  [trainer] Early stop at epoch {epoch} "
                f"(no improvement for {config.EARLY_STOP_PATIENCE} epochs)"
            )
            break

    epoch_bar.close()
    print("\n✓ Training complete.")
    return state