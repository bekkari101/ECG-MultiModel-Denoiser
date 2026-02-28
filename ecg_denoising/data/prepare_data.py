"""
prepare_data.py
===============
Pre-process raw CSV patient splits → windowed .npy arrays ready for training.

INPUT layout (your existing outputs folder):
    <SOURCE>/
        train/<patient_id>/patient_<id>_train.csv
        val/<patient_id>/patient_<id>_val.csv
        test/<patient_id>/patient_<id>_test.csv

OUTPUT layout written to DATA/ folder:
    DATA/
        phase1/              ← identity warmup  (clean → clean)
            train/  clean_<patient_id>.npy   shape (N, WINDOW_SIZE, N_LEADS)
            val/    clean_<patient_id>.npy
            test/   clean_<patient_id>.npy

        phase2/              ← denoising, LOW  noise (SNR_LOW  dB)
            train/  clean_<patient_id>.npy   ← clean targets
                    noisy_<patient_id>.npy   ← noisy inputs
            val/    …
            test/   …

        phase3/              ← denoising, MID  noise (SNR_MEDIUM dB)
            train/  clean_<patient_id>.npy
                    noisy_<patient_id>.npy
            …

        phase4/              ← denoising, HIGH noise (SNR_HIGH  dB)
            train/  clean_<patient_id>.npy
                    noisy_<patient_id>.npy
            …

USAGE:
    python prepare_data.py
    python prepare_data.py --source "C:/Users/user/Desktop/Work/ECG/outputs"
    python prepare_data.py --source /path/to/outputs --dest /path/to/DATA
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from data.windowing import extract_windows, prepare_leads


# ─────────────────────────────────────────────────────────────────────────────
# Default paths
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SOURCE = config.DATA_DIR
DEFAULT_DEST   = config.DATA_PROCESSED_DIR   # = BASE_DIR / "DATA"


# ─────────────────────────────────────────────────────────────────────────────
# Noise helper  (inline — no separate noise.py needed)
# ─────────────────────────────────────────────────────────────────────────────

def _add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to reach the target SNR (dB)."""
    signal_power = np.mean(signal ** 2)
    snr_linear   = 10 ** (snr_db / 10.0)
    noise_power  = signal_power / (snr_linear + 1e-12)
    noise        = rng.standard_normal(signal.shape).astype(np.float32) * np.sqrt(noise_power)
    return signal + noise


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(csv_path: str) -> np.ndarray:
    """Read patient CSV → float32 array (T, C) dropping non-signal columns."""
    df   = pd.read_csv(csv_path)
    drop = [c for c in df.columns if c.lower() in ("time", "patient_id")]
    df   = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
    return df.values.astype(np.float32)


def _find_csvs(source_dir: Path, split: str) -> list:
    """Scan <source_dir>/<split>/**/*.csv, detect all patients then apply filter."""
    pattern = str(source_dir / split / "**" / "*.csv")
    paths   = sorted(glob.glob(pattern, recursive=True))
    all_patients = [(Path(p).parent.name, p) for p in paths]
    
    print(f"    Detected {len(all_patients)} total patients: {[pid for pid, _ in all_patients]}")
    
    # Apply patient filter if configured
    if config.PATIENT_FILTER is not None:
        filtered_patients = [(pid, path) for pid, path in all_patients if pid in config.PATIENT_FILTER]
        print(f"    Filter: Using {len(filtered_patients)}/{len(all_patients)} patients (filter: {config.PATIENT_FILTER})")
        return filtered_patients
    else:
        print(f"    Using all {len(all_patients)} available patients")
        return all_patients


def _save(arr: np.ndarray, folder: Path, name: str):
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / name, arr)


# ─────────────────────────────────────────────────────────────────────────────
# Phase definitions
# ─────────────────────────────────────────────────────────────────────────────

# phase_num → (snr_db or None for clean-only)
PHASES = {
    1: None,              # identity warmup — clean only
    2: config.SNR_LOW,    # LOW  noise
    3: config.SNR_MEDIUM, # MID  noise
    4: config.SNR_HIGH,   # HIGH noise
}

PHASE_LABELS = {
    1: "Identity Warmup   (clean → clean)",
    2: f"LOW  noise        ({config.SNR_LOW} dB)",
    3: f"MID  noise        ({config.SNR_MEDIUM} dB)",
    4: f"HIGH noise        ({config.SNR_HIGH} dB)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-split processor
# ─────────────────────────────────────────────────────────────────────────────

def process_split(source_dir: Path, dest_dir: Path, split: str, rng: np.random.Generator):
    csv_list = _find_csvs(source_dir, split)

    if not csv_list:
        print(f"\n  [!] No CSVs found for split='{split}' under {source_dir / split}")
        return

    print(f"\n  {'─'*60}")
    print(f"  {split.upper()}  —  {len(csv_list)} patient file(s)")
    print(f"  {'─'*60}")

    # Output dirs for each phase
    phase_dirs = {p: dest_dir / f"phase{p}" / split for p in PHASES}

    total_windows = 0

    # ── Outer bar: patients ────────────────────────────────────────────────
    patient_bar = tqdm(
        csv_list,
        desc=f"    [{split}] patients",
        unit="patient",
        dynamic_ncols=True,
        colour="cyan",
        leave=True,
    )

    for patient_id, csv_path in patient_bar:
        patient_bar.set_postfix(patient=patient_id)

        # 1. Load CSV
        raw = _load_csv(csv_path)
        if raw.shape[0] < config.WINDOW_SIZE:
            tqdm.write(
                f"    [skip] {patient_id}: only {raw.shape[0]} samples "
                f"(need ≥ {config.WINDOW_SIZE})"
            )
            continue

        # 2. Ensure exactly N_LEADS channels
        sig = prepare_leads(raw, [f"ch{i}" for i in range(raw.shape[1])])

        # 3. Extract clean windows  →  (N, WINDOW_SIZE, N_LEADS)
        windows_clean = extract_windows(sig)
        N = len(windows_clean)
        if N == 0:
            tqdm.write(f"    [skip] {patient_id}: no complete windows extracted")
            continue

        # ── Inner bar: windows ────────────────────────────────────────────
        win_bar = tqdm(
            range(N),
            desc=f"      [{split}/{patient_id}] windows",
            unit="win",
            dynamic_ncols=True,
            colour="yellow",
            leave=False,
        )

        # Pre-allocate noisy buffers for phases 2/3/4
        noisy_bufs = {p: np.empty_like(windows_clean) for p in PHASES if PHASES[p] is not None}

        for i in win_bar:
            w = windows_clean[i]
            for phase_num, snr_db in PHASES.items():
                if snr_db is not None:
                    noisy_bufs[phase_num][i] = _add_noise(w, snr_db, rng)

        win_bar.close()

        # ── Save all phases for this patient ──────────────────────────────
        for phase_num, snr_db in PHASES.items():
            out_dir = phase_dirs[phase_num]
            # clean targets always saved (all phases need them)
            _save(windows_clean, out_dir, f"clean_{patient_id}.npy")
            # noisy inputs only for phases 2/3/4
            if snr_db is not None:
                _save(noisy_bufs[phase_num], out_dir, f"noisy_{patient_id}.npy")

        total_windows += N
        patient_bar.set_postfix(patient=patient_id, windows=N)
        tqdm.write(
            f"    ✓  {patient_id:<12s}  "
            f"raw={raw.shape[0]:>8,d} samples  "
            f"windows={N:>6,d}  "
            f"→  4 phases written"
        )

    patient_bar.close()
    tqdm.write(f"\n    Total windows [{split}]: {total_windows:,d}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(dest_dir: Path):
    print(f"\n{'═'*65}")
    print("  DATA FOLDER SUMMARY")
    print(f"{'═'*65}")
    print(f"  Root : {dest_dir}\n")

    for phase_num, label in PHASE_LABELS.items():
        phase_dir = dest_dir / f"phase{phase_num}"
        if not phase_dir.exists():
            continue
        print(f"  phase{phase_num}  [{label}]")
        for split in ("train", "val", "test"):
            split_dir = phase_dir / split
            if not split_dir.exists():
                continue
            npy_files = sorted(split_dir.glob("*.npy"))
            total = sum(np.load(p, mmap_mode="r").shape[0] for p in npy_files)
            print(f"    {split:<5s}  {len(npy_files):>3d} file(s)  {total:>10,d} windows")
        print()

    print(f"{'═'*65}")
    print("  ✓  Preparation done.  Run  python main.py  to start training.")
    print(f"{'═'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Config banner
# ─────────────────────────────────────────────────────────────────────────────

def print_config_banner(source_dir: Path, dest_dir: Path):
    print(f"\n{'═'*65}")
    print("  ECG Data Preparation Pipeline")
    print(f"{'═'*65}")
    print(f"  Source       : {source_dir}")
    print(f"  Destination  : {dest_dir}")
    print(f"  Window size  : {config.WINDOW_SIZE} samples  "
          f"({config.WINDOW_SECONDS} s @ {config.SAMPLING_FREQ} Hz)")
    print(f"  Window shift : {config.WINDOW_SHIFT} samples  ({config.SHIFT_SECONDS} s)")
    print(f"  N leads      : {config.N_LEADS}")
    print(f"  Seed         : {config.SEED}")
    print(f"  Phases:")
    for phase_num, label in PHASE_LABELS.items():
        print(f"    Phase {phase_num} — {label}")
    print(f"{'═'*65}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-process ECG CSV splits into windowed .npy arrays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", type=str, default=str(DEFAULT_SOURCE),
        help="Root folder containing train/val/test sub-directories with CSVs",
    )
    parser.add_argument(
        "--dest", type=str, default=str(DEFAULT_DEST),
        help="Output DATA folder where processed .npy files will be written",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    dest_dir   = Path(args.dest)

    print_config_banner(source_dir, dest_dir)

    if not source_dir.exists():
        print(f"\n  [ERROR] Source directory not found:\n    {source_dir}")
        print(f"  Use --source to provide the correct path.  Example:")
        print(f'    python prepare_data.py --source "C:/Users/user/Desktop/Work/ECG/outputs"')
        sys.exit(1)

    rng = np.random.default_rng(config.SEED)

    for split in ("train", "val", "test"):
        process_split(source_dir, dest_dir, split, rng)

    print_summary(dest_dir)


if __name__ == "__main__":
    main()