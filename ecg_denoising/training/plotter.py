"""
training/plotter.py — Non-interactive plots for Rotating Epoch Training.

Every point is colour-coded by epoch type (clean/low/mid/high).
Vertical tick marks on the x-axis show which type each epoch is.
All plots saved as PNG — training never pauses.
"""

import os, sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import config


# ─────────────────────────────────────────────────────────────────────────────
# Epoch-type colours and markers
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_COLOR = {
    "clean": "#27ae60",   # green
    "low":   "#3498db",   # blue
    "mid":   "#f39c12",   # orange
    "high":  "#e74c3c",   # red
}
_TYPE_MARKER = {
    "clean": "o",
    "low":   "s",
    "mid":   "^",
    "high":  "D",
}
_TYPE_LABEL = {
    "clean": "Clean (identity)",
    "low":   f"Low noise ({config.SNR_LOW} dB)",
    "mid":   f"Mid noise ({config.SNR_MEDIUM} dB)",
    "high":  f"High noise ({config.SNR_HIGH} dB)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _legend_patches():
    return [
        mpatches.Patch(color=_TYPE_COLOR[t], label=_TYPE_LABEL[t])
        for t in config.EPOCH_ROTATION
    ]


def _extract(history: list, split: str, key: str):
    """Return (epochs, values, epoch_types) for split/key."""
    epochs, vals, types = [], [], []
    for rec in history:
        ep    = rec.get("epoch", len(epochs) + 1)
        etype = rec.get("epoch_type", "clean")
        data  = rec.get(split, {})
        v = data.get(key)
        if v is not None:
            try:
                fv = float(v)
                if fv == fv:   # not NaN
                    epochs.append(ep)
                    vals.append(fv)
                    types.append(etype)
            except (TypeError, ValueError):
                pass
    return epochs, vals, types


def _draw_type_strip(ax, history):
    """
    Draw a thin coloured strip at the bottom of the axes showing epoch type.
    One rectangle per epoch, coloured by type.
    """
    for rec in history:
        ep    = rec.get("epoch", 1)
        etype = rec.get("epoch_type", "clean")
        color = _TYPE_COLOR.get(etype, "#cccccc")
        ax.axvspan(ep - 0.5, ep + 0.5,
                   ymin=0, ymax=0.02,
                   facecolor=color, alpha=0.9, zorder=3, clip_on=False)


def _scatter_by_type(ax, epochs, vals, types, markersize=5, alpha=0.85,
                     linestyle="-", linewidth=1.4, connect=True):
    """
    Plot points individually coloured by epoch type, with connecting lines.
    """
    if not epochs:
        return
    # Draw connecting line first (grey, behind coloured dots)
    if connect and len(epochs) > 1:
        ax.plot(epochs, vals, color="#cccccc", linewidth=linewidth * 0.7,
                linestyle=linestyle, alpha=0.5, zorder=1)
    # Draw coloured points
    for ep, val, etype in zip(epochs, vals, types):
        ax.plot(ep, val,
                marker=_TYPE_MARKER.get(etype, "o"),
                color=_TYPE_COLOR.get(etype, "#999999"),
                markersize=markersize,
                markeredgecolor="white",
                markeredgewidth=0.4,
                linestyle="",
                alpha=alpha,
                zorder=2)


def _save_fig(fig, filename: str):
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    path = config.PLOT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _base_fig(title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Epoch  (colour = noise type, shape = noise type)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Individual plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss(history: list):
    """Train + Val loss, each point coloured by epoch type."""
    fig, ax = _base_fig("Train / Val Loss  (per epoch type)", "Combined Loss")

    for split, linestyle, label_suffix in [("train", "-", "Train"), ("val", "--", "Val")]:
        epochs, vals, types = _extract(history, split, "loss")
        if connect := len(epochs) > 1:
            ax.plot(epochs, vals, color="#bbbbbb",
                    linewidth=1.0, linestyle=linestyle, alpha=0.4, zorder=1)
        for ep, val, etype in zip(epochs, vals, types):
            ax.plot(ep, val,
                    marker=_TYPE_MARKER.get(etype, "o"),
                    color=_TYPE_COLOR.get(etype, "#999"),
                    markersize=5 if split == "train" else 7,
                    markeredgecolor="white" if split == "train" else _TYPE_COLOR.get(etype, "#999"),
                    markeredgewidth=0.5 if split == "train" else 1.5,
                    linestyle="", alpha=0.9, zorder=2)

    # Legend: type colours + train/val marker size distinction
    patches = _legend_patches()
    train_proxy = plt.Line2D([], [], color="grey", lw=1,   linestyle="-",  label="Train (small dot)")
    val_proxy   = plt.Line2D([], [], color="grey", lw=1.5, linestyle="--", label="Val   (large dot)")
    ax.legend(handles=patches + [train_proxy, val_proxy], fontsize=7, loc="upper right")
    _draw_type_strip(ax, history)
    _save_fig(fig, "train_loss.png")


def plot_val_mse(history: list):
    """Validation MSE coloured by epoch type."""
    fig, ax = _base_fig("Validation MSE", "MSE")
    epochs, vals, types = _extract(history, "val", "mse")
    _scatter_by_type(ax, epochs, vals, types)
    ax.legend(handles=_legend_patches(), fontsize=7)
    _draw_type_strip(ax, history)
    _save_fig(fig, "valid_loss.png")


def plot_snr_improvement(history: list):
    """Val SNR improvement — negative values highlighted in red."""
    fig, ax = _base_fig(
        "Val SNR Improvement (dB)  — negative = signal worsened",
        "SNR Improvement (dB)"
    )
    epochs, vals, types = _extract(history, "val", "snr_improve")
    if epochs:
        _scatter_by_type(ax, epochs, vals, types)
        ax.axhline(0, color="black", linewidth=0.9, linestyle=":", alpha=0.6)
        # Shade where negative
        ep_arr  = np.array(epochs)
        val_arr = np.array(vals)
        neg_mask = val_arr < 0
        if neg_mask.any():
            ax.fill_between(ep_arr, val_arr, 0,
                            where=neg_mask, interpolate=True,
                            alpha=0.18, color="red", label="Signal worsened")
    ax.legend(handles=_legend_patches() + [
        mpatches.Patch(color="red", alpha=0.3, label="SNR < 0 (signal worse)")
    ], fontsize=7)
    _draw_type_strip(ax, history)
    _save_fig(fig, "snr_improvement.png")


def plot_prd(history: list):
    fig, ax = _base_fig("Val PRD (%)  ↓ better", "PRD (%)")
    epochs, vals, types = _extract(history, "val", "prd")
    _scatter_by_type(ax, epochs, vals, types)
    ax.legend(handles=_legend_patches(), fontsize=7)
    _draw_type_strip(ax, history)
    _save_fig(fig, "prd.png")


def plot_pearson(history: list):
    fig, ax = _base_fig("Pearson Correlation ρ  ↑ better  (target ≥ 0.99)", "ρ")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="#27ae60", linewidth=0.8, linestyle=":", alpha=0.6)

    for split, linestyle in [("train", "-"), ("val", "--")]:
        epochs, vals, types = _extract(history, split, "pearson")
        if len(epochs) > 1:
            ax.plot(epochs, vals, color="#cccccc",
                    linewidth=0.8, linestyle=linestyle, alpha=0.4, zorder=1)
        for ep, val, etype in zip(epochs, vals, types):
            ax.plot(ep, val,
                    marker=_TYPE_MARKER.get(etype, "o"),
                    color=_TYPE_COLOR.get(etype, "#999"),
                    markersize=5 if split == "train" else 7,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    linestyle="", alpha=0.9, zorder=2)

    ax.legend(handles=_legend_patches() + [
        plt.Line2D([], [], color="grey", lw=1, linestyle="-",  label="Train"),
        plt.Line2D([], [], color="grey", lw=1, linestyle="--", label="Val"),
    ], fontsize=7, loc="lower right")
    _draw_type_strip(ax, history)
    _save_fig(fig, "correlation.png")


def plot_lr(history: list):
    fig, ax = _base_fig("Learning Rate Schedule", "LR (log scale)")
    epochs, lrs = [], []
    for rec in history:
        ep = rec.get("epoch", 1)
        lr = rec.get("lr")
        if lr is not None:
            epochs.append(ep)
            lrs.append(float(lr))
    if epochs:
        ax.semilogy(epochs, lrs, "o-", color="#8e44ad", linewidth=1.5,
                    markersize=4, alpha=0.85)
    ax.set_ylabel("Learning Rate (log)")
    _draw_type_strip(ax, history)
    _save_fig(fig, "learning_rate.png")


def plot_rotation_summary(history: list):
    """
    4-panel grid: one panel per epoch type, showing val loss for epochs of that type only.
    Lets you see how well the model is learning each noise level independently.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes_flat = axes.flatten()
    rotation  = config.EPOCH_ROTATION   # ["clean", "low", "mid", "high"]

    for ax, etype in zip(axes_flat, rotation):
        color = _TYPE_COLOR[etype]
        # Filter val loss for this epoch type only
        epochs, vals, _ = _extract(history, "val", "loss")
        ep_f = [e for e, t in zip(epochs, [r.get("epoch_type","clean") for r in history]) if t == etype]
        vl_f = [v for v, t in zip(vals,   [r.get("epoch_type","clean") for r in history]) if t == etype]
        if ep_f:
            ax.plot(ep_f, vl_f, "o-", color=color,
                    linewidth=1.5, markersize=5, alpha=0.85)
            ax.scatter(ep_f, vl_f, color=color, s=30, zorder=3)
        ax.set_title(f"{_TYPE_LABEL[etype]}  (val loss)", fontsize=9)
        ax.set_xlabel("Global epoch")
        ax.set_ylabel("Val loss")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Val Loss per Epoch Type  (independent view)", fontsize=12, y=1.01)
    plt.tight_layout()
    _save_fig(fig, "rotation_summary.png")


def plot_metrics_grid(history: list):
    """4-panel: MSE / MAE / RMSE / Huber for val set, coloured by epoch type."""
    keys  = [("mse", "MSE"), ("mae", "MAE"), ("rmse", "RMSE"), ("huber", "Huber Loss")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for ax, (key, title) in zip(axes.flatten(), keys):
        epochs, vals, types = _extract(history, "val", key)
        _scatter_by_type(ax, epochs, vals, types)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        _draw_type_strip(ax, history)

    fig.suptitle("Validation Metrics  (colour = epoch type)", fontsize=12, y=1.01)
    plt.tight_layout()
    _save_fig(fig, "metrics_grid.png")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def update_plots(history: list):
    """Re-render all plots. Called after every epoch."""
    if not history:
        return
    plot_loss(history)
    plot_val_mse(history)
    plot_snr_improvement(history)
    plot_prd(history)
    plot_pearson(history)
    plot_lr(history)
    plot_rotation_summary(history)
    plot_metrics_grid(history)