"""
model/lstm_model.py — Stacked LSTM ECG Denoiser.

Architecture:
  Input  : (batch, WINDOW_SIZE, N_LEADS)   →   (batch, 1440, 2)  at 360 Hz / 4 s
  LSTM   : N stacked layers, hidden sizes from config.LSTM_LAYERS
  Linear : project LSTM output → N_LEADS
  Output : (batch, WINDOW_SIZE, N_LEADS)

Run this file directly to print the full model + config summary.
"""

import os, sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import torch
import torch.nn as nn
import config


class ECGDenoisingLSTM(nn.Module):
    """Stacked (optionally Bidirectional) LSTM with inter-layer dropout — seq2seq."""

    def __init__(
        self,
        input_size:    int   = config.N_LEADS,
        output_size:   int   = config.N_LEADS,
        hidden_sizes:  list  = config.LSTM_LAYERS,
        dropout:       float = config.LSTM_DROPOUT,
        bidirectional: bool  = config.BIDIRECTIONAL,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.n_directions  = 2 if bidirectional else 1

        self.lstm_layers    = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i, h in enumerate(hidden_sizes):
            in_sz   = input_size if i == 0 else hidden_sizes[i - 1] * self.n_directions
            is_last = (i == len(hidden_sizes) - 1)

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=in_sz,
                    hidden_size=h,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )
            if not is_last:
                self.dropout_layers.append(nn.Dropout(p=dropout))

        lstm_out_size = hidden_sizes[-1] * self.n_directions
        self.fc = nn.Linear(lstm_out_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (batch, seq_len, input_size)
        Returns:
            out : (batch, seq_len, output_size)
        """
        out = x
        for i, lstm in enumerate(self.lstm_layers):
            out, _ = lstm(out)
            if i < len(self.dropout_layers):
                out = self.dropout_layers[i](out)
        return self.fc(out)


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module):
    total    = count_parameters(model)
    sep      = "═" * 65
    thin_sep = "─" * 65

    print(f"\n{sep}")
    print(f"  ECG Denoising LSTM  —  Model & Configuration Summary")
    print(sep)

    # ── Architecture ──────────────────────────────────────────────
    print("  ARCHITECTURE")
    print(thin_sep)
    n_dir = 2 if config.BIDIRECTIONAL else 1
    for i, h in enumerate(config.LSTM_LAYERS):
        in_sz  = config.N_LEADS if i == 0 else config.LSTM_LAYERS[i - 1] * n_dir
        out_sz = h * n_dir
        print(f"    LSTM layer {i+1} :  in={in_sz:>4d}  hidden={h:>4d}  "
              f"out={out_sz:>4d}  bidir={config.BIDIRECTIONAL}")
        if i < len(config.LSTM_LAYERS) - 1:
            print(f"    Dropout {i+1}     :  p={config.LSTM_DROPOUT}")
    fc_in = config.LSTM_LAYERS[-1] * n_dir
    print(f"    Linear (FC)   :  in={fc_in:>4d}  out={config.N_LEADS:>4d}")
    print(f"\n    Input  tensor :  (batch, {config.WINDOW_SIZE}, {config.N_LEADS})")
    print(f"    Output tensor :  (batch, {config.WINDOW_SIZE}, {config.N_LEADS})")
    print(f"\n    ★ Trainable parameters : {total:,}")

    # ── Signal / Windowing ────────────────────────────────────────
    print(f"\n{thin_sep}")
    print("  SIGNAL & WINDOWING")
    print(thin_sep)
    print(f"    Sampling freq  : {config.SAMPLING_FREQ} Hz")
    print(f"    Window size    : {config.SAMPLING_FREQ} Hz × {config.WINDOW_SECONDS} s "
          f"= {config.WINDOW_SIZE} samples")
    print(f"    Window shift   : {config.SAMPLING_FREQ} Hz × {config.SHIFT_SECONDS} s "
          f"= {config.WINDOW_SHIFT} samples  "
          f"({100*(1 - config.WINDOW_SHIFT/config.WINDOW_SIZE):.1f}% overlap)")
    print(f"    Leads (N)      : {config.N_LEADS}")

    # ── Noise ─────────────────────────────────────────────────────
    print(f"\n{thin_sep}")
    print("  NOISE AUGMENTATION")
    print(thin_sep)
    phase_labels = {
        "low":    f"Phase 2  LOW  noise : {config.SNR_LOW:>3d} dB",
        "medium": f"Phase 3  MID  noise : {config.SNR_MEDIUM:>3d} dB",
        "high":   f"Phase 4  HIGH noise : {config.SNR_HIGH:>3d} dB",
    }
    for key, label in phase_labels.items():
        print(f"    {label}")

    # ── Training ──────────────────────────────────────────────────
    print(f"\n{thin_sep}")
    print("  TRAINING — 4 PHASES")
    print(thin_sep)
    print(f"    Device         : {config.DEVICE}")
    if config.DEVICE == "cpu":
        print(f"    CPU threads    : {config.CPU_THREAD_LIMIT}  (capped)")
    print(f"    Batch size     : {config.BATCH_SIZE}")
    print(f"    Seed           : {config.SEED}")
    print(f"    Phase 1 epochs : {config.PHASE1_EPOCHS:>3d}  (identity warmup,     LR={config.PHASE1_LR})")
    print(f"    Phase 2 epochs : {config.PHASE2_EPOCHS:>3d}  (LOW  noise, {config.SNR_LOW} dB,  LR={config.PHASE2_LR})")
    print(f"    Phase 3 epochs : {config.PHASE3_EPOCHS:>3d}  (MID  noise, {config.SNR_MEDIUM} dB,  LR={config.PHASE3_LR})")
    print(f"    Phase 4 epochs : {config.PHASE4_EPOCHS:>3d}  (HIGH noise,  {config.SNR_HIGH} dB,  LR={config.PHASE4_LR})")
    print(f"    Grad clip      : {config.GRAD_CLIP}")
    print(f"    Loss weights   : MSE×{config.MSE_WEIGHT}  +  MAE×{config.MAE_WEIGHT}")
    print(f"    Scheduler      : ReduceLROnPlateau  "
          f"patience={config.SCHEDULER_PATIENCE}  "
          f"factor={config.SCHEDULER_FACTOR}  "
          f"min_lr={config.SCHEDULER_MIN_LR}")
    es = config.EARLY_STOP_PATIENCE
    print(f"    Early stop     : {'disabled' if es is None else f'patience={es}'}")

    # ── Paths ─────────────────────────────────────────────────────
    print(f"\n{thin_sep}")
    print("  PATHS")
    print(thin_sep)
    print(f"    Data dir       : {config.DATA_DIR}")
    print(f"    Processed data : {config.DATA_PROCESSED_DIR}")
    print(f"    Checkpoints    : {config.CHECKPOINT_DIR}")
    print(f"    Plots          : {config.PLOT_DIR}")
    print(f"    Training state : {config.TRAINING_STATE_PATH}")
    print(sep + "\n")


# ─────────────────────────────────────────────
# Run as script → summary + forward smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = ECGDenoisingLSTM()
    print_model_summary(model)

    dummy = torch.randn(4, config.WINDOW_SIZE, config.N_LEADS)
    out   = model(dummy)
    print(f"  Forward pass smoke-test  ✓")
    print(f"    Input  shape : {tuple(dummy.shape)}")
    print(f"    Output shape : {tuple(out.shape)}")
    print(f"    Max abs value: {out.abs().max().item():.6f}")