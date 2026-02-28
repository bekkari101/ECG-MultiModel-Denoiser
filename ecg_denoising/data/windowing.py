"""
data/windowing.py — Sliding-window extraction from ECG time-series.

Window size  : WINDOW_SIZE  samples  (global, from config)
Window shift : WINDOW_SHIFT samples  (global, from config)
"""

import os, sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import numpy as np
import config


def extract_windows(signal: np.ndarray) -> np.ndarray:
    """
    Slice a continuous ECG signal into overlapping windows.

    Args:
        signal : np.ndarray shape (T, N_LEADS) — full recording or split segment

    Returns:
        windows : np.ndarray shape (N_windows, WINDOW_SIZE, N_LEADS) float32
    """
    T, n_leads = signal.shape
    ws = config.WINDOW_SIZE
    sh = config.WINDOW_SHIFT

    if T < ws:
        return np.empty((0, ws, n_leads), dtype=np.float32)

    starts  = range(0, T - ws + 1, sh)
    windows = np.stack([signal[s: s + ws] for s in starts], axis=0)
    return windows.astype(np.float32)


def prepare_leads(signal: np.ndarray, sig_names: list[str]) -> np.ndarray:
    """
    Ensure signal has exactly N_LEADS channels.
    If only 1 lead exists it is duplicated; extras are dropped.

    Args:
        signal    : shape (T, C)
        sig_names : list of channel names (used only for logging)

    Returns:
        shape (T, N_LEADS) float32
    """
    T, C = signal.shape
    target = config.N_LEADS

    if C >= target:
        out = signal[:, :target]
    else:
        # duplicate last channel to fill up to N_LEADS
        repeats = [signal[:, i % C : i % C + 1] for i in range(target)]
        out = np.concatenate(repeats, axis=1)
        print(f"[windowing] Only {C} lead(s) found ({sig_names}); duplicating to {target}.")

    return out.astype(np.float32)
