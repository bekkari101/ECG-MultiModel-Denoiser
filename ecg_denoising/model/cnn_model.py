"""
model/cnn_model.py — 1D CNN ECG Denoiser.

Architecture:
  Input  : (batch, WINDOW_SIZE, N_LEADS)   →   (batch, 1440, 2)  at 360 Hz / 4 s
  CNN    : 1D convolutions with residual connections
  Output : (batch, WINDOW_SIZE, N_LEADS)

Uses 1D convolutions suitable for time-series data.
"""

import os, sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class ResidualBlock1D(nn.Module):
    """Residual block for 1D CNN."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ECGDenoisingCNN(nn.Module):
    """1D CNN with residual connections for ECG denoising."""
    
    def __init__(
        self,
        input_channels: int = config.N_LEADS,
        output_channels: int = config.N_LEADS,
        layers: list = config.CNN_LAYERS,
        kernel_size: int = config.CNN_KERNEL_SIZE,
        pool_size: int = config.CNN_POOL_SIZE,
    ):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, layers[0], kernel_size=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.residual_blocks.append(
                ResidualBlock1D(layers[i], layers[i+1], kernel_size)
            )
            if i < len(layers) - 2:  # Don't pool after last block
                self.residual_blocks.append(nn.MaxPool1d(pool_size))
        
        # Output projection
        self.output_proj = nn.Conv1d(layers[-1], output_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (batch, seq_len, input_channels)
        Returns:
            out : (batch, seq_len, output_channels)
        """
        # Input: (batch, seq_len, channels) → (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Residual blocks with pooling
        for i, block in enumerate(self.residual_blocks):
            if isinstance(block, ResidualBlock1D):
                x = block(x)
            else:  # Pooling layer
                x = block(x)
        
        # Upsample back to original length
        x = F.interpolate(x, size=x.shape[2] * (config.CNN_POOL_SIZE ** 2), mode='linear', align_corners=False)
        
        # Output projection
        x = self.output_proj(x)
        
        # Output: (batch, channels, seq_len) → (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Ensure correct length
        if x.shape[1] != config.WINDOW_SIZE:
            x = F.interpolate(x.transpose(1, 2), size=config.WINDOW_SIZE, mode='linear', align_corners=False).transpose(1, 2)
        
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module):
    total = count_parameters(model)
    print(f"\n{'='*65}")
    print(f"  ECG Denoising CNN  —  Model Summary")
    print(f"{'='*65}")
    print(f"  Architecture: 1D CNN with residual connections")
    print(f"  Layers: {config.CNN_LAYERS}")
    print(f"  Kernel size: {config.CNN_KERNEL_SIZE}")
    print(f"  Pool size: {config.CNN_POOL_SIZE}")
    print(f"  Trainable parameters: {total:,}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    model = ECGDenoisingCNN()
    print_model_summary(model)
    
    dummy = torch.randn(4, config.WINDOW_SIZE, config.N_LEADS)
    out = model(dummy)
    print(f"  Forward pass smoke-test  ✓")
    print(f"    Input  shape : {tuple(dummy.shape)}")
    print(f"    Output shape : {tuple(out.shape)}")
