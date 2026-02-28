"""
training/logger.py — JSON state logger for Rotating Epoch Training.

Persists:
  - current_epoch  (last completed epoch number)
  - best_val_loss
  - no_improve     (early stopping counter)
  - full history   (one record per epoch, including epoch_type)
  - lr_history
"""

import json
import os
import shutil
from pathlib import Path

import sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import config

STATE_PATH  = config.TRAINING_STATE_PATH
BACKUP_PATH = Path(str(STATE_PATH) + ".bak")


def _default_state() -> dict:
    return {
        "current_epoch": 0,
        "best_val_loss": float("inf"),
        "no_improve":    0,
        "lr_history":    [],
        "history":       [],
    }


def load_state() -> dict:
    for path in (STATE_PATH, BACKUP_PATH):
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                state = json.load(f)
            assert "history" in state and isinstance(state["history"], list)
            print(f"  [logger] Resumed from {path}")
            return state
        except Exception as e:
            print(f"  [logger] State corrupted ({e}), trying backup …")
    print("  [logger] Starting fresh.")
    return _default_state()


def save_state(state: dict):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    tmp = Path(str(STATE_PATH) + ".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        if os.path.exists(STATE_PATH):
            shutil.copy2(STATE_PATH, BACKUP_PATH)
        os.replace(tmp, STATE_PATH)
    except Exception as e:
        print(f"  [logger] WARNING: Could not save — {e}")


def _clean_metrics(d: dict) -> dict:
    """Sanitise floats for JSON (NaN → None, round to 8 dp)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = None if v != v else round(v, 8)
        else:
            out[k] = v
    return out


def append_epoch(
    state: dict,
    epoch: int,
    epoch_type: str,
    train_metrics: dict,
    val_metrics: dict,
    lr: float,
) -> dict:
    """Append one epoch record and persist."""
    record = {
        "epoch":      epoch,
        "epoch_type": epoch_type,
        "lr":         lr,
        "train":      _clean_metrics(train_metrics),
        "val":        _clean_metrics(val_metrics),
    }
    state["history"].append(record)
    state["current_epoch"] = epoch
    state["lr_history"].append(lr)
    save_state(state)
    return state