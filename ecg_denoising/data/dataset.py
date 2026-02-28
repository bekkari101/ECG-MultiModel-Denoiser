"""
data/dataset.py — ECG Dataset loaders for Rotating Epoch Training.

Folder layout expected (produced by prepare_data.py):
    DATA/
        phase1/train/   clean_<id>.npy
        phase2/train/   clean_<id>.npy  noisy_<id>.npy   (LOW  noise)
        phase3/train/   clean_<id>.npy  noisy_<id>.npy   (MID  noise)
        phase4/train/   clean_<id>.npy  noisy_<id>.npy   (HIGH noise)
        (same for val/)

Public API used by trainer:
    get_all_loaders() → {
        "clean": {"train": DataLoader, "val": DataLoader},
        "low":   {"train": DataLoader, "val": DataLoader},
        "mid":   {"train": DataLoader, "val": DataLoader},
        "high":  {"train": DataLoader, "val": DataLoader},
    }

Legacy phase loaders kept for backward compat with prepare_data.py.
"""

import os, sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

import config


# ─────────────────────────────────────────────────────────────────────────────
# File loading
# ─────────────────────────────────────────────────────────────────────────────

def _count_windows(fp: str) -> int:
    return np.load(fp, mmap_mode="r").shape[0]


def load_npy_files(file_paths: list, desc: str = "Loading") -> np.ndarray:
    if not file_paths:
        raise RuntimeError(f"[dataset] No files to load for: {desc}")
    counts = [_count_windows(fp) for fp in file_paths]
    total  = sum(counts)
    arrays = []
    with tqdm(total=total, desc=f"  {desc}", unit="win",
              dynamic_ncols=True, colour="blue") as bar:
        for fp, n in zip(file_paths, counts):
            arrays.append(np.load(fp).astype(np.float32))
            bar.update(n)
    return np.concatenate(arrays, axis=0)


def _phase_split_dir(phase: int, split: str) -> Path:
    folder = config.DATA_PROCESSED_DIR / f"phase{phase}" / split
    if not folder.exists():
        raise FileNotFoundError(
            f"Data folder not found: {folder}\n"
            f"Run  python prepare_data.py  first."
        )
    return folder


def _get_files(phase: int, split: str, prefix: str) -> list:
    folder = _phase_split_dir(phase, split)
    files  = sorted(folder.glob(f"{prefix}_*.npy"))
    if not files:
        raise FileNotFoundError(
            f"No '{prefix}_*.npy' files in {folder}\n"
            f"Run  python prepare_data.py  first."
        )
    # PATIENT_FILTER=None means load everything
    if config.PATIENT_FILTER is not None:
        files = [f for f in files
                 if f.stem.split('_', 1)[1] in config.PATIENT_FILTER]
        print(f"    Filter: {len(files)} {prefix} files (phase{phase}/{split})")
    else:
        print(f"    Loading all {len(files)} {prefix} files (phase{phase}/{split})")
    if not files:
        raise FileNotFoundError(
            f"No files matched PATIENT_FILTER={config.PATIENT_FILTER} in {folder}"
        )
    return [str(f) for f in files]


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class CleanECGDataset(Dataset):
    """clean → clean (identity)."""
    def __init__(self, windows: np.ndarray):
        self.w = torch.from_numpy(windows)
    def __len__(self): return len(self.w)
    def __getitem__(self, i): return self.w[i], self.w[i]


class NoisyECGDataset(Dataset):
    """noisy input → clean target."""
    def __init__(self, noisy: np.ndarray, clean: np.ndarray):
        assert len(noisy) == len(clean)
        self.noisy = torch.from_numpy(noisy)
        self.clean = torch.from_numpy(clean)
    def __len__(self): return len(self.clean)
    def __getitem__(self, i): return self.noisy[i], self.clean[i]


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size  = config.BATCH_SIZE,
        shuffle     = shuffle,
        num_workers = config.NUM_WORKERS,
        pin_memory  = config.PIN_MEMORY,
    )


def _clean_loaders(split: str) -> DataLoader:
    files   = _get_files(phase=1, split=split, prefix="clean")
    windows = load_npy_files(files, desc=f"clean [{split}]")
    return _make_loader(CleanECGDataset(windows), shuffle=(split == "train"))


def _noisy_loaders(phase: int, label: str, split: str) -> DataLoader:
    clean_files = _get_files(phase=phase, split=split, prefix="clean")
    noisy_files = _get_files(phase=phase, split=split, prefix="noisy")
    clean_win   = load_npy_files(clean_files, desc=f"{label} clean [{split}]")
    noisy_win   = load_npy_files(noisy_files, desc=f"{label} noisy [{split}]")
    return _make_loader(NoisyECGDataset(noisy_win, clean_win), shuffle=(split == "train"))


# ─────────────────────────────────────────────────────────────────────────────
# Public API — used by trainer
# ─────────────────────────────────────────────────────────────────────────────

def get_all_loaders() -> dict:
    """
    Load all four noise-type loaders.

    Returns:
        {
            "clean": {"train": DataLoader, "val": DataLoader},
            "low":   {"train": DataLoader, "val": DataLoader},
            "mid":   {"train": DataLoader, "val": DataLoader},
            "high":  {"train": DataLoader, "val": DataLoader},
        }
    """
    loaders = {}

    print("\n[dataset] Loading clean (identity) …")
    loaders["clean"] = {
        "train": _clean_loaders("train"),
        "val":   _clean_loaders("val"),
    }

    for noise_type, phase, snr in [
        ("low",  2, config.SNR_LOW),
        ("mid",  3, config.SNR_MEDIUM),
        ("high", 4, config.SNR_HIGH),
    ]:
        print(f"\n[dataset] Loading {noise_type} noise ({snr} dB) …")
        loaders[noise_type] = {
            "train": _noisy_loaders(phase, noise_type, "train"),
            "val":   _noisy_loaders(phase, noise_type, "val"),
        }
        n_train = len(loaders[noise_type]["train"].dataset)
        n_val   = len(loaders[noise_type]["val"].dataset)
        print(f"           train={n_train:,}  val={n_val:,} windows")

    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# Legacy phase loaders (used by prepare_data.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_phase1_loaders() -> dict:
    return {s: _clean_loaders(s) for s in ("train", "val")}

def get_phase2_loaders() -> dict:
    return {s: _noisy_loaders(2, "low",  s) for s in ("train", "val")}

def get_phase3_loaders() -> dict:
    return {s: _noisy_loaders(3, "mid",  s) for s in ("train", "val")}

def get_phase4_loaders() -> dict:
    return {s: _noisy_loaders(4, "high", s) for s in ("train", "val")}