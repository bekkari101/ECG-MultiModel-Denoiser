"""
training/checkpoint.py — Save and load model + optimizer checkpoints.
"""

import os, sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import torch
import config


LAST_PATH = config.CHECKPOINT_DIR / "last.pt"
BEST_PATH = config.CHECKPOINT_DIR / "best.pt"


def save_checkpoint(model, optimizer, epoch: int, phase: int, loss: float, is_best: bool = False):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    state = {
        "epoch":     epoch,
        "phase":     phase,
        "loss":      loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, LAST_PATH)
    if is_best:
        torch.save(state, BEST_PATH)
        print(f"  [ckpt] ★ Best checkpoint saved  (loss={loss:.6f})")


def load_checkpoint(model, optimizer, path=None, device=None):
    """
    Load checkpoint. Returns (epoch, phase, loss) or (0, 1, inf) if not found.
    """
    path = path or LAST_PATH
    device = device or torch.device(config.DEVICE)

    if not os.path.exists(path):
        print(f"  [ckpt] No checkpoint at {path} — starting fresh.")
        return 0, 1, float("inf")

    try:
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        epoch = ckpt.get("epoch", 0)
        phase = ckpt.get("phase", 1)
        loss  = ckpt.get("loss",  float("inf"))
        print(f"  [ckpt] Resumed from epoch={epoch}  phase={phase}  loss={loss:.6f}")
        return epoch, phase, loss
    except Exception as e:
        print(f"  [ckpt] !! Checkpoint corrupted ({e}). Trying best.pt ...")
        if path != BEST_PATH and os.path.exists(BEST_PATH):
            return load_checkpoint(model, optimizer, BEST_PATH, device)
        print("  [ckpt] No valid checkpoint found — starting fresh.")
        return 0, 1, float("inf")
