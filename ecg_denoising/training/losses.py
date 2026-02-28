"""
training/losses.py — Combined MSE + MAE loss and helper metric losses.
"""

import os, sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class CombinedLoss(nn.Module):
    """Primary training loss: MSE + λ·MAE."""

    def __init__(self, mse_w: float = config.MSE_WEIGHT, mae_w: float = config.MAE_WEIGHT):
        super().__init__()
        self.mse_w = mse_w
        self.mae_w = mae_w
        self.mse   = nn.MSELoss()
        self.mae   = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse_w * self.mse(pred, target) + self.mae_w * self.mae(pred, target)
