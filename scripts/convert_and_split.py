
"""Select N random MIT-BIH records, convert to CSV, and split to train/test/val.

Usage examples:
  python scripts/convert_and_split.py
  python scripts/convert_and_split.py --data-dir raw/mit-bih-arrhythmia-database-1.0.0 --n 2

Outputs written to `outputs/` by default.
"""
import os
import random
from pathlib import Path

# Configuration (no CLI args — change these globals as needed)
# DATA_DIR defaults to the workspace `raw/mit-bih-arrhythmia-database-1.0.0` folder
# using an absolute path based on the repository root (one level above `scripts`).
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = str(BASE_DIR / 'raw' / 'mit-bih-arrhythmia-database-1.0.0')
N = 10      # Increased to process more patients for better training
SEED = 42
OUTPUT_DIR = str(BASE_DIR / 'outputs')

import numpy as np
import pandas as pd

try:
    import wfdb
except Exception:
    wfdb = None

# We split time-series by contiguous ranges (no random splitting), so no sklearn needed


def list_records(data_dir):
    records_file = os.path.join(data_dir, 'RECORDS')
    if not os.path.exists(records_file):
        raise FileNotFoundError(f"RECORDS file not found in {data_dir}")
    with open(records_file, 'r', encoding='utf8') as f:
        recs = [ln.strip() for ln in f if ln.strip()]
    return recs


def read_record(data_dir, rec_name):
    path = os.path.join(data_dir, rec_name)
    # wfdb.rdsamp accepts a path without extension
    # older/newer wfdb versions have different signatures; call without
    # the `physical` kwarg for maximum compatibility and handle return types below.
    record = wfdb.rdsamp(path)
    # wfdb.rdsamp historically returned (signals, fields) but newer APIs
    # may return a Record object. Handle both.
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


def to_dataframe(signals, sig_names, fs, patient_id):
    n = signals.shape[0]
    time = np.arange(n) / fs
    cols = ['time'] + list(sig_names)
    arr = np.column_stack([time, signals])
    df = pd.DataFrame(arr, columns=cols)
    df['patient_id'] = str(patient_id)
    return df


def split_and_save(df_all, out_dir, seed=42):
    # This function is no longer used. Keep as placeholder in case combined
    # splitting is desired in the future.
    raise RuntimeError('split_and_save is deprecated; use per-patient splitting')


def split_and_save_patient(df, out_dir, rec_name, train_frac=0.7, test_frac=0.2, val_frac=0.1):
    """Split a single patient's dataframe by contiguous time ranges and save files.

    The split uses the dataframe's existing order (time ascending). Files written:
      patient_<rec>_train.csv, patient_<rec>_test.csv, patient_<rec>_val.csv
    """
    n = len(df)
    if n == 0:
        return None, None, None
    # compute split indices
    train_end = int(n * train_frac)
    test_end = train_end + int(n * test_frac)
    # ensure full coverage: val takes the rest
    train_df = df.iloc[0:train_end]
    test_df = df.iloc[train_end:test_end]
    val_df = df.iloc[test_end:]

    # Create per-split per-patient directories: out_dir/train/<rec>/, out_dir/test/<rec>/, out_dir/val/<rec>/
    train_dir = os.path.join(out_dir, 'train', str(rec_name))
    test_dir = os.path.join(out_dir, 'test', str(rec_name))
    val_dir = os.path.join(out_dir, 'val', str(rec_name))
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_path = os.path.join(train_dir, f'patient_{rec_name}_train.csv')
    test_path = os.path.join(test_dir, f'patient_{rec_name}_test.csv')
    val_path = os.path.join(val_dir, f'patient_{rec_name}_val.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(val_path, index=False)

    return train_path, test_path, val_path


def main():
    # Use module-level globals instead of CLI args
    if wfdb is None:
        raise RuntimeError('The `wfdb` package is required. Install from requirements.txt and try again.')

    data_dir = DATA_DIR
    recs = list_records(data_dir)
    if N > len(recs):
        raise ValueError(f'Requested n={N} > available records {len(recs)}')

    random.seed(SEED)
    chosen = random.sample(recs, N)
    out_dir = OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    for rec in chosen:
        print(f'Reading {rec}...')
        signals, sig_names, fs = read_record(data_dir, rec)
        df = to_dataframe(signals, sig_names, fs, rec)
        patient_csv = os.path.join(out_dir, f'patient_{rec}.csv')
        df.to_csv(patient_csv, index=False)
        print('  wrote', patient_csv)

        # Split this patient's data by contiguous time ranges and save separately
        train_path, test_path, val_path = split_and_save_patient(df, out_dir, rec_name=rec)
        print('  wrote splits:')
        print('    ', train_path)
        print('    ', test_path)
        print('    ', val_path)


if __name__ == '__main__':
    main()
