"""
training/metrics.py — Evaluation metrics for ECG denoising.

All functions accept torch Tensors (pred, target) on any device.
"""

import torch
import torch.nn.functional as F


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(pred, target).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(pred, target).item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(pred, target).sqrt().item()


def huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> float:
    return F.huber_loss(pred, target, delta=delta).item()


def prd(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Percent Root-mean-square Difference (%)."""
    num = ((pred - target) ** 2).sum()
    den = (target ** 2).sum()
    return (100.0 * (num / (den + 1e-12)).sqrt()).item()


def snr_improvement(pred: torch.Tensor, target: torch.Tensor, noisy: torch.Tensor | None = None) -> float:
    """
    SNR improvement in dB.
    If noisy is None, SNR of pred vs target is returned directly.
    Otherwise: SNR(pred) - SNR(noisy).
    """
    def _snr(signal, noise):
        s_pow = (signal ** 2).mean()
        n_pow = (noise ** 2).mean()
        return (10 * torch.log10(s_pow / (n_pow + 1e-12))).item()

    snr_pred = _snr(target, pred - target)
    if noisy is None:
        return snr_pred
    snr_noisy = _snr(target, noisy - target)
    return snr_pred - snr_noisy


def pearson_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Pearson correlation across batch × leads."""
    p = pred.reshape(-1)
    t = target.reshape(-1)
    p_mean = p.mean(); t_mean = t.mean()
    num = ((p - p_mean) * (t - t_mean)).sum()
    den = ((p - p_mean)**2).sum().sqrt() * ((t - t_mean)**2).sum().sqrt()
    return (num / (den + 1e-12)).item()


def compute_all(pred: torch.Tensor, target: torch.Tensor, noisy: torch.Tensor | None = None) -> dict:
    """Return dict of all metrics."""
    return {
        "mse":         mse(pred, target),
        "mae":         mae(pred, target),
        "rmse":        rmse(pred, target),
        "huber":       huber(pred, target),
        "prd":         prd(pred, target),
        "snr_improve": snr_improvement(pred, target, noisy),
        "pearson":     pearson_correlation(pred, target),
    }
