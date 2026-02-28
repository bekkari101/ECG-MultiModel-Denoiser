"""
prepare_data_unified.py
========================
Combined ECG data preparation pipeline that processes raw MIT-BIH data 
directly into windowed .npy arrays ready for training.

INPUT layout (raw MIT-BIH data):
    <RAW_DATA_DIR>/
        *.atr, *.dat, *.hea files (MIT-BIH format)
        RECORDS file listing available records

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
    python prepare_data_unified.py
    python prepare_data_unified.py --raw-data-dir "C:/Users/user/Desktop/Work/ECG/raw/mit-bih-arrhythmia-database-1.0.0"
    python prepare_data_unified.py --dest /path/to/DATA --n-patients 10
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from data.windowing import extract_windows, prepare_leads

try:
    import wfdb
except Exception:
    wfdb = None

# ─────────────────────────────────────────────────────────────────────────────
# Default paths
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_RAW_DATA_DIR = Path(__file__).resolve().parents[2] / 'raw' / 'mit-bih-arrhythmia-database-1.0.0'
DEFAULT_DEST = config.DATA_PROCESSED_DIR   # = BASE_DIR / "DATA"

# ─────────────────────────────────────────────────────────────────────────────
# MIT-BIH data reading functions
# ─────────────────────────────────────────────────────────────────────────────

def list_records(data_dir):
    """List all available records from RECORDS file."""
    records_file = data_dir / 'RECORDS'
    if not records_file.exists():
        raise FileNotFoundError(f"RECORDS file not found in {data_dir}")
    with open(records_file, 'r', encoding='utf8') as f:
        recs = [ln.strip() for ln in f if ln.strip()]
    return recs

def read_record(data_dir, rec_name):
    """Read a single MIT-BIH record."""
    path = data_dir / rec_name
    record = wfdb.rdsamp(str(path))
    
    # Handle different wfdb versions
    if isinstance(record, tuple) and len(record) == 2:
        signals, meta = record
        fields = meta
    else:
        signals = getattr(record, 'p_signals', None) or getattr(record, 'd_signal', None)
        fields = record.__dict__ if hasattr(record, '__dict__') else {}

    if signals is None:
        raise RuntimeError('Unable to read signals for ' + rec_name)

    sig_names = fields.get('sig_name') or ['ch1', 'ch2']
    fs = fields.get('fs', 360)
    return np.asarray(signals), sig_names, float(fs)

def split_and_save_patient(signals, sig_names, fs, rec_name, train_frac=0.7, test_frac=0.2, val_frac=0.1):
    """Split a single patient's signals by contiguous time ranges."""
    n = signals.shape[0]
    if n == 0:
        return None, None, None
    
    # Compute split indices
    train_end = int(n * train_frac)
    test_end = train_end + int(n * test_frac)
    
    # Split the data
    train_signals = signals[0:train_end]
    test_signals = signals[train_end:test_end]
    val_signals = signals[test_end:]
    
    return train_signals, test_signals, val_signals

# ─────────────────────────────────────────────────────────────────────────────
# Noise helper
# ─────────────────────────────────────────────────────────────────────────────

def _add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to reach the target SNR (dB)."""
    signal_power = np.mean(signal ** 2)
    snr_linear   = 10 ** (snr_db / 10.0)
    noise_power  = signal_power / (snr_linear + 1e-12)
    noise        = rng.standard_normal(signal.shape).astype(np.float32) * np.sqrt(noise_power)
    return signal + noise

# ─────────────────────────────────────────────────────────────────────────────
# Windowing and saving functions
# ─────────────────────────────────────────────────────────────────────────────

def process_signals_to_windows(signals, rec_name, rng):
    """Process raw signals into windows for all phases."""
    # Ensure exactly N_LEADS channels
    sig = prepare_leads(signals, [f"ch{i}" for i in range(signals.shape[1])])
    
    # Extract clean windows
    windows_clean = extract_windows(sig)
    N = len(windows_clean)
    if N == 0:
        return None, None
    
    # Generate noisy versions for phases 2/3/4
    noisy_bufs = {}
    for phase_num, snr_db in {2: config.SNR_LOW, 3: config.SNR_MEDIUM, 4: config.SNR_HIGH}.items():
        noisy_bufs[phase_num] = np.empty_like(windows_clean)
        for i in range(N):
            noisy_bufs[phase_num][i] = _add_noise(windows_clean[i], snr_db, rng)
    
    return windows_clean, noisy_bufs

def _save(arr: np.ndarray, folder: Path, name: str):
    """Save numpy array to file."""
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / name, arr)

# ─────────────────────────────────────────────────────────────────────────────
# Phase definitions
# ─────────────────────────────────────────────────────────────────────────────

PHASES = {
    1: None,              # identity warmup — clean only
    2: config.SNR_LOW,    # LOW  noise
    3: config.SNR_MEDIUM, # MID  noise
    4: config.SNR_HIGH,   # HIGH noise
}

PHASE_LABELS = {
    1: "Identity Warmup   (clean -> clean)",
    2: f"LOW  noise        ({config.SNR_LOW} dB)",
    3: f"MID  noise        ({config.SNR_MEDIUM} dB)",
    4: f"HIGH noise        ({config.SNR_HIGH} dB)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Main processing function
# ─────────────────────────────────────────────────────────────────────────────

def process_all_patients(raw_data_dir, dest_dir, n_patients, rng):
    """Process all patients from raw MIT-BIH data."""
    # List available records
    all_records = list_records(raw_data_dir)
    print(f"Found {len(all_records)} records in MIT-BIH database")
    
    # Select patients
    if n_patients <= 0:
        # Process all available patients
        n_patients = len(all_records)
    
    if n_patients > len(all_records):
        n_patients = len(all_records)
    
    # Use first N patients for reproducibility (or random sample)
    # MODIFIED: Process all patients by default, ignore PATIENT_FILTER for comprehensive data preparation
    if hasattr(config, 'PATIENT_FILTER') and config.PATIENT_FILTER and n_patients <= 10:
        # Only use filter if explicitly requesting small number of patients
        selected_patients = [rec for rec in all_records if rec in config.PATIENT_FILTER]
        print(f"Using filtered patients: {selected_patients}")
    else:
        # Process all available patients
        selected_patients = all_records[:n_patients]
        print(f"Using first {len(selected_patients)} patients: {selected_patients[:5]}{'...' if len(selected_patients) > 5 else ''}")
        print(f"Total patients to process: {len(selected_patients)}")
    
    # Create output directories for each phase and split
    phase_dirs = {}
    for phase_num in PHASES:
        for split in ['train', 'val', 'test']:
            phase_dirs[(phase_num, split)] = dest_dir / f"phase{phase_num}" / split
    
    # Process each patient
    total_windows = {split: 0 for split in ['train', 'val', 'test']}
    
    patient_bar = tqdm(
        selected_patients,
        desc="Processing patients",
        unit="patient",
        dynamic_ncols=True,
        colour="cyan",
    )
    
    for rec_name in patient_bar:
        patient_bar.set_postfix(patient=rec_name)
        
        try:
            # Read raw MIT-BIH record
            signals, sig_names, fs = read_record(raw_data_dir, rec_name)
            
            # Split into train/val/test
            train_signals, test_signals, val_signals = split_and_save_patient(signals, sig_names, fs, rec_name)
            
            # Process each split
            for split_name, split_signals in [('train', train_signals), ('val', val_signals), ('test', test_signals)]:
                if split_signals is None or len(split_signals) == 0:
                    continue
                
                # Convert to windows
                windows_clean, noisy_bufs = process_signals_to_windows(split_signals, rec_name, rng)
                
                if windows_clean is None or len(windows_clean) == 0:
                    continue
                
                # Save all phases for this split
                for phase_num, snr_db in PHASES.items():
                    out_dir = phase_dirs[(phase_num, split_name)]
                    
                    # Save clean targets (all phases need them)
                    _save(windows_clean, out_dir, f"clean_{rec_name}.npy")
                    
                    # Save noisy inputs (only for phases 2/3/4)
                    if snr_db is not None and noisy_bufs:
                        _save(noisy_bufs[phase_num], out_dir, f"noisy_{rec_name}.npy")
                
                total_windows[split_name] += len(windows_clean)
            
            patient_bar.set_postfix(
                patient=rec_name, 
                train_windows=len(windows_clean) if 'train_signals' in locals() and train_signals is not None else 0
            )
            
        except Exception as e:
            tqdm.write(f"    [ERROR] Failed to process {rec_name}: {str(e)}")
            continue
    
    patient_bar.close()
    return total_windows

# ─────────────────────────────────────────────────────────────────────────────
# Summary and config functions
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(dest_dir, total_windows):
    """Print summary of processed data."""
    print(f"\n{'='*65}")
    print("  DATA FOLDER SUMMARY")
    print(f"{'='*65}")
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

    print(f"{'='*65}")
    print("  SUCCESS: Preparation done.  Run  python main.py  to start training.")
    print(f"{'='*65}\n")

def print_config_banner(raw_data_dir, dest_dir, n_patients):
    """Print configuration banner."""
    print(f"\n{'='*65}")
    print("  UNIFIED ECG Data Preparation Pipeline")
    print(f"{'='*65}")
    print(f"  Raw Data Dir : {raw_data_dir}")
    print(f"  Destination  : {dest_dir}")
    if n_patients <= 0:
        print(f"  N Patients   : ALL (will process all available patients)")
    else:
        print(f"  N Patients   : {n_patients}")
    print(f"  Window size  : {config.WINDOW_SIZE} samples  "
          f"({config.WINDOW_SECONDS} s @ {config.SAMPLING_FREQ} Hz)")
    print(f"  Window shift : {config.WINDOW_SHIFT} samples  ({config.SHIFT_SECONDS} s)")
    print(f"  N leads      : {config.N_LEADS}")
    print(f"  Seed         : {config.SEED}")
    print(f"  Phases:")
    for phase_num, label in PHASE_LABELS.items():
        print(f"    Phase {phase_num} — {label}")
    print(f"{'='*65}")

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified ECG data preparation: raw MIT-BIH → windowed .npy arrays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-data-dir", type=str, default=str(DEFAULT_RAW_DATA_DIR),
        help="Directory containing raw MIT-BIH files (.atr, .dat, .hea, RECORDS)",
    )
    parser.add_argument(
        "--dest", type=str, default=str(DEFAULT_DEST),
        help="Output DATA folder where processed .npy files will be written",
    )
    parser.add_argument(
        "--n-patients", type=int, default=0,
        help="Number of patients to process (use all available if 0 or not specified)",
    )
    args = parser.parse_args()

    raw_data_dir = Path(args.raw_data_dir)
    dest_dir = Path(args.dest)

    print_config_banner(raw_data_dir, dest_dir, args.n_patients)

    if wfdb is None:
        raise RuntimeError('The `wfdb` package is required. Install from requirements.txt and try again.')

    if not raw_data_dir.exists():
        print(f"\n  [ERROR] Raw data directory not found:\n    {raw_data_dir}")
        sys.exit(1)

    # Set random seed
    random.seed(config.SEED)
    rng = np.random.default_rng(config.SEED)

    # Process all patients
    total_windows = process_all_patients(raw_data_dir, dest_dir, args.n_patients, rng)

    # Print summary
    print_summary(dest_dir, total_windows)

if __name__ == "__main__":
    main()
