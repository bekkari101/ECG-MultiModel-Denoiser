"""
model/transformer_model.py — Transformer-based ECG Denoiser.

Architecture:
  Input  : (batch, WINDOW_SIZE, N_LEADS)   →   (batch, 1440, 2)  at 360 Hz / 4 s
  Transformer : Multi-head self-attention layers
  Output : (batch, WINDOW_SIZE, N_LEADS)

Uses transformer architecture for sequence modeling.
"""

import os, sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import torch
import torch.nn as nn
import math
import config


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class ECGDenoisingTransformer(nn.Module):
    """Transformer-based ECG denoiser."""
    
    def __init__(
        self,
        input_size: int = config.N_LEADS,
        output_size: int = config.N_LEADS,
        d_model: int = config.TRANSFORMER_D_MODEL,
        nhead: int = config.TRANSFORMER_NHEAD,
        num_layers: int = config.TRANSFORMER_NUM_LAYERS,
        dropout: float = config.TRANSFORMER_DROPOUT,
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (batch, seq_len, input_size)
        Returns:
            out : (batch, seq_len, output_size)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer layers
        x = self.transformer(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module):
    total = count_parameters(model)
    print(f"\n{'='*65}")
    print(f"  ECG Denoising Transformer  —  Model Summary")
    print(f"{'='*65}")
    print(f"  Architecture: Multi-head self-attention")
    print(f"  D_model: {config.TRANSFORMER_D_MODEL}")
    print(f"  Heads: {config.TRANSFORMER_NHEAD}")
    print(f"  Layers: {config.TRANSFORMER_NUM_LAYERS}")
    print(f"  Dropout: {config.TRANSFORMER_DROPOUT}")
    print(f"  Trainable parameters: {total:,}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    model = ECGDenoisingTransformer()
    print_model_summary(model)
    
    dummy = torch.randn(4, config.WINDOW_SIZE, config.N_LEADS)
    out = model(dummy)
    print(f"  Forward pass smoke-test  ✓")
    print(f"    Input  shape : {tuple(dummy.shape)}")
    print(f"    Output shape : {tuple(out.shape)}")
