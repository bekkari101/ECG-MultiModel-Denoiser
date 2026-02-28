"""
data/noise.py — Composite noise augmentation for ECG windows.

Noise model:
    x_noisy = x_clean + gaussian + baseline_wander + powerline + emg

SNR is enforced per-window.
"""

import os, sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import numpy as np
import config


def _signal_power(x: np.ndarray) -> float:
    """Mean power of signal (averaged over all samples & channels)."""
    p = np.mean(x ** 2)
    return float(p) if p > 0 else 1e-12


def _add_gaussian(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_power = _signal_power(x)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), x.shape)
    return noise.astype(np.float32)


def _add_baseline_wander(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency sinusoid (0.05–0.5 Hz) baseline wander."""
    n_samples, n_leads = x.shape
    sig_power = _signal_power(x)
    noise_power = sig_power / (10 ** (snr_db / 10)) * 0.5   # partial contribution
    t = np.arange(n_samples) / config.SAMPLING_FREQ
    noise = np.zeros_like(x)
    for ch in range(n_leads):
        freq = rng.uniform(0.05, 0.5)
        phase = rng.uniform(0, 2 * np.pi)
        amp = np.sqrt(2 * noise_power)
        noise[:, ch] = (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
    return noise.astype(np.float32)


def _add_powerline(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """50 Hz or 60 Hz powerline interference."""
    n_samples, n_leads = x.shape
    sig_power = _signal_power(x)
    noise_power = sig_power / (10 ** (snr_db / 10)) * 0.3
    t = np.arange(n_samples) / config.SAMPLING_FREQ
    freq = rng.choice([50.0, 60.0])
    noise = np.zeros_like(x)
    for ch in range(n_leads):
        phase = rng.uniform(0, 2 * np.pi)
        amp = np.sqrt(2 * noise_power)
        noise[:, ch] = (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
    return noise.astype(np.float32)


def _add_emg(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """High-frequency EMG / muscle artefact (bandpass-like white noise)."""
    sig_power = _signal_power(x)
    noise_power = sig_power / (10 ** (snr_db / 10)) * 0.2
    noise = rng.normal(0, np.sqrt(noise_power), x.shape).astype(np.float32)
    # crude high-pass: diff to emphasise HF content
    noise[1:] = noise[1:] - 0.5 * noise[:-1]
    return noise.astype(np.float32)


def add_composite_noise(
    window_clean: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Apply composite noise to a single ECG window.

    Args:
        window_clean : np.ndarray shape (WINDOW_SIZE, N_LEADS)  float32
        snr_db       : target SNR in dB
        rng          : numpy Generator for reproducibility

    Returns:
        noisy window, same shape as input, float32
    """
    if rng is None:
        rng = np.random.default_rng()

    noise  = _add_gaussian(window_clean, snr_db, rng)
    noise += _add_baseline_wander(window_clean, snr_db, rng)
    noise += _add_powerline(window_clean, snr_db, rng)
    noise += _add_emg(window_clean, snr_db, rng)

    return (window_clean + noise).astype(np.float32)


def augment_windows(
    windows_clean: np.ndarray,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    For every clean window generate 3 noisy versions (low / medium / high SNR).

    Args:
        windows_clean : shape (N, WINDOW_SIZE, N_LEADS) float32
        rng           : numpy Generator

    Returns:
        dict with keys "low", "medium", "high" each shape (N, WINDOW_SIZE, N_LEADS)
    """
    if rng is None:
        rng = np.random.default_rng(config.SEED)

    result = {}
    for level, snr_db in config.NOISE_LEVELS.items():
        noisy = np.stack(
            [add_composite_noise(w, snr_db, rng) for w in windows_clean],
            axis=0,
        )
        result[level] = noisy
    return result
