"""
model/rnn_model.py — GRU-based ECG Denoiser.

Architecture:
  Input  : (batch, WINDOW_SIZE, N_LEADS)   →   (batch, 1440, 2)  at 360 Hz / 4 s
  GRU    : Stacked GRU layers with dropout
  Output : (batch, WINDOW_SIZE, N_LEADS)

Uses GRU instead of LSTM for faster training and potentially better performance.
"""

import os, sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import torch
import torch.nn as nn
import config


class ECGDenoisingRNN(nn.Module):
    """Stacked GRU with dropout for ECG denoising."""
    
    def __init__(
        self,
        input_size: int = config.N_LEADS,
        output_size: int = config.N_LEADS,
        hidden_sizes: list = config.RNN_LAYERS,
        dropout: float = config.RNN_DROPOUT,
        bidirectional: bool = config.BIDIRECTIONAL,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1
        
        self.gru_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i, h in enumerate(hidden_sizes):
            in_sz = input_size if i == 0 else hidden_sizes[i - 1] * self.n_directions
            is_last = (i == len(hidden_sizes) - 1)
            
            self.gru_layers.append(
                nn.GRU(
                    input_size=in_sz,
                    hidden_size=h,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )
            if not is_last:
                self.dropout_layers.append(nn.Dropout(p=dropout))
        
        gru_out_size = hidden_sizes[-1] * self.n_directions
        self.fc = nn.Linear(gru_out_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (batch, seq_len, input_size)
        Returns:
            out : (batch, seq_len, output_size)
        """
        out = x
        for i, gru in enumerate(self.gru_layers):
            out, _ = gru(out)
            if i < len(self.dropout_layers):
                out = self.dropout_layers[i](out)
        return self.fc(out)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module):
    total = count_parameters(model)
    print(f"\n{'='*65}")
    print(f"  ECG Denoising RNN (GRU)  —  Model Summary")
    print(f"{'='*65}")
    print(f"  Architecture: Stacked GRU layers")
    print(f"  Layers: {config.RNN_LAYERS}")
    print(f"  Bidirectional: {config.BIDIRECTIONAL}")
    print(f"  Dropout: {config.RNN_DROPOUT}")
    print(f"  Trainable parameters: {total:,}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    model = ECGDenoisingRNN()
    print_model_summary(model)
    
    dummy = torch.randn(4, config.WINDOW_SIZE, config.N_LEADS)
    out = model(dummy)
    print(f"  Forward pass smoke-test  ✓")
    print(f"    Input  shape : {tuple(dummy.shape)}")
    print(f"    Output shape : {tuple(out.shape)}")
