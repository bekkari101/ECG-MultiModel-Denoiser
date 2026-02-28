"""
tools/ecg_tester.py — ECG Denoising Test Tool (Wizard Edition)
===============================================================
Multi-step Tkinter wizard replaces the old single-dialog + argparse approach.

Steps:
  1  Select Model          — best / last / custom .pt checkpoint
  2  Select Data Source    — Mode A (prepared DATA/), B (CSV), C (raw WFDB)
  3  Select Patient        — auto-populated or entered manually
  4  Configure Test        — SNR, display windows, skip-model toggle
  5  Processing            — animated progress + live log
  6  Results               — interactive matplotlib viewer embedded in Tk

Bug fixed: matplotlib CheckButtons.rectangles → use labels loop instead
           (`.rectangles` was removed in matplotlib ≥ 3.7)
"""

import os
import sys
import threading
import time
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Button as MplButton, CheckButtons

# ── project root ──────────────────────────────────────────────────────────────
_root_path = Path(__file__).resolve().parents[1]
if str(_root_path) not in sys.path:
    sys.path.insert(0, str(_root_path))

import config
from model.model_factory import create_model
from data.windowing      import extract_windows, prepare_leads

# All supported model types
SUPPORTED_MODELS = ["lstm", "cnn", "rnn", "transformer", "embedding"]


# ═════════════════════════════════════════════════════════════════════════════
# Theme constants
# ═════════════════════════════════════════════════════════════════════════════
BG       = "#0f172a"
BG2      = "#1e293b"
BG3      = "#0d1b2e"
FG       = "#e2e8f0"
FG_DIM   = "#94a3b8"
ACCENT   = "#38bdf8"
ACCENT2  = "#818cf8"
GREEN    = "#34d399"
RED      = "#f87171"
ENTRY_BG = "#0f1f35"
BTN_BG   = "#1e3a5f"
SEP      = "#1e3a5f"

FONT      = ("Segoe UI", 9)
FONT_B    = ("Segoe UI", 9, "bold")
FONT_H    = ("Segoe UI", 13, "bold")
FONT_SH   = ("Segoe UI", 10, "bold")
FONT_MONO = ("Consolas", 8)

PLOT_COLORS = {
    "clean":    "#00d4ff",
    "noisy":    "#ff6b35",
    "denoised": "#39ff88",
}
PLOT_LABELS = {
    "clean":    "Clean (ground truth)",
    "noisy":    "Noisy input",
    "denoised": "Denoised (model output)",
}


# ═════════════════════════════════════════════════════════════════════════════
# Backend helpers
# ═════════════════════════════════════════════════════════════════════════════

def _read_wfdb_record(raw_dir: Path, record: str):
    try:
        import wfdb
    except ImportError:
        raise ImportError("Install wfdb:  pip install wfdb")
    rec = wfdb.rdsamp(str(raw_dir / record))
    if isinstance(rec, tuple):
        signals, meta = rec
    else:
        signals = getattr(rec, "p_signals", None) or getattr(rec, "d_signal", None)
        meta    = rec.__dict__ if hasattr(rec, "__dict__") else {}
    sig_names = meta.get("sig_name", ["ch1", "ch2"])
    fs        = float(meta.get("fs", 360))
    return np.asarray(signals, dtype=np.float32), sig_names, fs


def _csv_to_array(csv_path: Path) -> np.ndarray:
    df   = pd.read_csv(csv_path)
    drop = [c for c in df.columns if c.lower() in ("time", "patient_id")]
    df   = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
    return df.values.astype(np.float32)


def _split_array(arr: np.ndarray, train=0.7, test=0.2, val=0.1):
    n     = len(arr)
    t_end = int(n * train)
    v_end = t_end + int(n * test)
    return {"train": arr[:t_end], "test": arr[t_end:v_end], "val": arr[v_end:]}


def _add_noise(clean: np.ndarray, snr_db: float) -> np.ndarray:
    sig_p = np.mean(clean ** 2)
    n_p   = sig_p / (10 ** (snr_db / 10.0) + 1e-12)
    return (clean + np.random.randn(*clean.shape).astype(np.float32) * np.sqrt(n_p))


def _compute_snr(pred: np.ndarray, target: np.ndarray) -> float:
    s = np.mean(target ** 2)
    n = np.mean((pred - target) ** 2) + 1e-12
    return float(10 * np.log10(s / n))


def _compute_prd(pred: np.ndarray, target: np.ndarray) -> float:
    return float(100 * np.sqrt(np.sum((pred - target) ** 2) / (np.sum(target ** 2) + 1e-12)))


def _auto_patients(split: str) -> list:
    folder = config.DATA_PROCESSED_DIR / "phase1" / split
    if not folder.exists():
        return []
    return [f.stem.replace("clean_", "") for f in sorted(folder.glob("clean_*.npy"))]


def _auto_csv_patients(csv_dir: str, split: str) -> list:
    folder = Path(csv_dir) / split
    if not folder.exists():
        return []
    return [d.name for d in sorted(folder.iterdir()) if d.is_dir()]


def _auto_ckpt_info(path: Path) -> str:
    """Return short summary string from a checkpoint file."""
    if not path.exists():
        return "not found"
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        e = ckpt.get("epoch", "?")
        p = ckpt.get("phase", "?")
        l = ckpt.get("loss", float("nan"))
        mt = _detect_model_type_from_state_dict(ckpt.get("model", {}))
        mt_str = f"  model={mt}" if mt else ""
        return f"epoch={e}  phase={p}  loss={l:.5f}{mt_str}"
    except Exception:
        return "unreadable"


def _detect_model_type_from_path(ckpt_path: str) -> str | None:
    """Detect model type from checkpoint path (e.g. outputs/lstm1/checkpoints/best.pt)."""
    path = Path(ckpt_path)
    for part in path.parts:
        for mt in SUPPORTED_MODELS:
            if part.lower().startswith(mt):
                return mt
    return None


def _detect_model_type_from_state_dict(state_dict: dict) -> str | None:
    """Detect model type by inspecting state dict key patterns."""
    key_str = " ".join(state_dict.keys())
    if "patch_encoder" in key_str or "vq_codebook" in key_str:
        return "embedding"
    if "gru_layers" in key_str:
        return "rnn"
    if "lstm_layers" in key_str:
        return "lstm"
    if "pos_encoder" in key_str and "transformer" in key_str:
        return "transformer"
    if "residual_blocks" in key_str:
        return "cnn"
    return None


def _scan_model_sessions() -> list:
    """
    Scan ecg_denoising/outputs/ for model session directories.
    Returns list of dicts: {name, path, model_type, best_exists, last_exists}.
    """
    outputs_dir = config.BASE_DIR / "outputs"
    if not outputs_dir.exists():
        return []
    sessions = []
    for item in sorted(outputs_dir.iterdir()):
        if not item.is_dir():
            continue
        ckpt_dir = item / "checkpoints"
        if not ckpt_dir.exists():
            continue
        # Detect model type from folder name
        model_type = None
        for mt in SUPPORTED_MODELS:
            if item.name.lower().startswith(mt):
                model_type = mt
                break
        sessions.append({
            "name":        item.name,
            "path":        ckpt_dir,
            "model_type":  model_type or "unknown",
            "best_exists": (ckpt_dir / "best.pt").exists(),
            "last_exists": (ckpt_dir / "last.pt").exists(),
        })
    return sessions


# ─── Data acquisition ─────────────────────────────────────────────────────────

def get_clean_windows_from_prepared(patient, split, log):
    path = config.DATA_PROCESSED_DIR / "phase1" / split / f"clean_{patient}.npy"
    if not path.exists():
        raise FileNotFoundError(f"No prepared data: {path}\nRun prepare_data.py first.")
    arr = np.load(path).astype(np.float32)
    log(f"  Loaded {path.name}  shape={arr.shape}")
    return arr


def get_clean_windows_from_prepared_with_noisy(patient, split, log):
    """Load clean + all 3 noise levels if they exist."""
    base = config.DATA_PROCESSED_DIR
    result = {}
    for phase, key in [(1,"clean"),(2,"low"),(3,"medium"),(4,"high")]:
        p = base / f"phase{phase}" / split
        # for phase1 it's clean_, for others noisy_
        fname = f"clean_{patient}.npy" if phase == 1 else f"noisy_{patient}.npy"
        fp = p / fname
        if fp.exists():
            arr = np.load(fp).astype(np.float32)
            result[key] = arr
            log(f"  Loaded {fname}  shape={arr.shape}")
    return result


def get_clean_windows_from_csv(csv_dir, patient, split, log):
    candidates = [
        Path(csv_dir) / split / patient / f"patient_{patient}_{split}.csv",
        Path(csv_dir) / f"patient_{patient}.csv",
    ]
    csv_path = next((c for c in candidates if c.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"CSV not found for patient={patient} split={split} under {csv_dir}."
        )
    raw     = _csv_to_array(csv_path)
    sig     = prepare_leads(raw, [f"ch{i}" for i in range(raw.shape[1])])
    windows = extract_windows(sig)
    log(f"  Loaded {csv_path.name}  raw={raw.shape}  windows={windows.shape}")
    return windows


def get_clean_windows_from_wfdb(raw_dir, record, split, log):
    log(f"  Reading WFDB record {record} from {raw_dir}")
    signals, sig_names, fs = _read_wfdb_record(Path(raw_dir), record)
    splits  = _split_array(signals)
    arr     = splits.get(split, splits["test"])
    sig     = prepare_leads(arr, sig_names)
    windows = extract_windows(sig)
    log(f"  Converted fs={fs}  split={split}  raw={arr.shape}  windows={windows.shape}")
    return windows


# ─── Inference ────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device, log):
    """Load any model type from checkpoint, auto-detecting architecture."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Auto-detect model type from path first, then state dict keys
    model_type = _detect_model_type_from_path(str(path))
    if model_type is None:
        model_type = _detect_model_type_from_state_dict(ckpt.get("model", {}))
    if model_type is None:
        model_type = config.MODEL_TYPE.lower()
        log(f"  [warn] Could not auto-detect model type, using config default: {model_type}")
    else:
        log(f"  [info] Detected model type: {model_type.upper()}")

    model = create_model(model_type).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    e = ckpt.get("epoch", "?")
    p = ckpt.get("phase", "?")
    l = ckpt.get("loss", float("nan"))
    log(f"  Model loaded  epoch={e}  phase={p}  val_loss={l:.6f}")
    log(f"  Architecture : {model_type.upper()}")
    return model, model_type


@torch.no_grad()
def run_inference(model, windows_clean, snr_db, device, batch_size, log):
    """
    Memory-safe streaming inference.

    Strategy:
      - Build noisy + denoised arrays in small batches, writing directly into
        pre-allocated float32 arrays.  Never holds more than one batch of
        intermediate tensors in memory at a time.
      - If a batch still OOMs, halve batch_size and retry automatically
        (down to a minimum of 1).
      - Warns and caps N at MAX_WINDOWS if the dataset is very large.
    """
    N = len(windows_clean)

    W = windows_clean.shape[1]
    L = windows_clean.shape[2]

    log(f"  Running inference  N={N}  SNR={snr_db} dB  batch={batch_size} …")

    # Pre-allocate output arrays (these are the only two large allocations)
    noisy    = np.empty((N, W, L), dtype=np.float32)
    denoised = np.empty((N, W, L), dtype=np.float32)

    # Build noisy array in chunks to avoid a single huge np.stack call
    NOISE_CHUNK = 256
    for s in range(0, N, NOISE_CHUNK):
        e = min(s + NOISE_CHUNK, N)
        for i in range(s, e):
            noisy[i] = _add_noise(windows_clean[i], snr_db)

    bs = batch_size
    start = 0
    while start < N:
        end = min(start + bs, N)
        try:
            batch = torch.from_numpy(noisy[start:end]).to(device)
            pred  = model(batch)
            # Handle embedding model returning (output, commit_loss) tuple
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = pred.cpu().numpy()
            denoised[start:end] = pred
            del batch, pred
            log(f"    {end}/{N} windows processed", replace_last=True)
            start = end
        except RuntimeError as exc:
            if "memory" in str(exc).lower() and bs > 1:
                bs = max(1, bs // 2)
                log(f"  [OOM] Reducing batch size to {bs} and retrying …")
            else:
                raise

    return {"clean": windows_clean, "noisy": noisy, "denoised": denoised}


def compute_metrics(results):
    clean    = results["clean"]
    noisy    = results["noisy"]
    denoised = results.get("denoised", None)
    m = {
        "snr_noisy":    _compute_snr(noisy, clean),
        "prd_noisy":    _compute_prd(noisy, clean),
        "mse_noisy":    float(np.mean((noisy - clean) ** 2)),
        "snr_denoised": float("nan"),
        "prd_denoised": float("nan"),
        "mse_denoised": float("nan"),
        "snr_improve":  float("nan"),
    }
    if denoised is not None:
        m["snr_denoised"] = _compute_snr(denoised, clean)
        m["prd_denoised"] = _compute_prd(denoised, clean)
        m["mse_denoised"] = float(np.mean((denoised - clean) ** 2))
        m["snr_improve"]  = m["snr_denoised"] - m["snr_noisy"]
    return m


# ═════════════════════════════════════════════════════════════════════════════
# Reusable UI widgets
# ═════════════════════════════════════════════════════════════════════════════

def styled_button(parent, text, command, accent=False, danger=False, **kw):
    bg = ACCENT if accent else (RED if danger else BTN_BG)
    fg = BG if accent else FG
    padx = kw.pop("padx", 12)
    pady = kw.pop("pady", 6)
    return tk.Button(
        parent, text=text, command=command,
        font=FONT_B, bg=bg, fg=fg,
        relief="flat", bd=0, cursor="hand2",
        padx=padx, pady=pady,
        activebackground=ACCENT2, activeforeground=FG,
        **kw,
    )


def section_label(parent, text):
    tk.Label(parent, text=text, font=FONT_SH, bg=BG, fg=ACCENT,
             anchor="w").pack(fill="x", pady=(14, 4))
    tk.Frame(parent, bg=SEP, height=1).pack(fill="x", pady=(0, 8))


def card(parent, **kw) -> tk.Frame:
    f = tk.Frame(parent, bg=BG2, bd=0, relief="flat", **kw)
    f.pack(fill="x", padx=0, pady=4)
    return f


def card_inner(parent) -> tk.Frame:
    f = tk.Frame(parent, bg=BG2)
    f.pack(fill="x", padx=14, pady=8)
    return f


def radio_row(parent, text, variable, value, command=None):
    rb = tk.Radiobutton(
        parent, text=text, variable=variable, value=value,
        font=FONT, bg=BG2, fg=FG, selectcolor=BG,
        activebackground=BG2, activeforeground=ACCENT,
        command=command,
    )
    rb.pack(anchor="w", pady=2)
    return rb


def lbl(parent, text, dim=False, bold=False, mono=False, **kw):
    f = FONT_MONO if mono else (FONT_B if bold else FONT)
    c = FG_DIM if dim else FG
    return tk.Label(parent, text=text, font=f, bg=BG2, fg=c, anchor="w", **kw)


def entry_field(parent, var, width=26, state="normal"):
    return tk.Entry(
        parent, textvariable=var,
        bg=ENTRY_BG, fg=FG, insertbackground=FG,
        relief="flat", bd=4, font=FONT, width=width, state=state,
    )


def dark_combo(parent, var, values, width=20):
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("D.TCombobox",
                    fieldbackground=ENTRY_BG, background=BG2,
                    foreground=FG, selectbackground=BG2,
                    arrowcolor=ACCENT)
    return ttk.Combobox(parent, textvariable=var, values=values,
                        state="readonly", style="D.TCombobox",
                        font=FONT, width=width)


# ═════════════════════════════════════════════════════════════════════════════
# Step pages  (each is a tk.Frame with a .build() and optional .validate())
# ═════════════════════════════════════════════════════════════════════════════

class StepBase(tk.Frame):
    title = ""
    subtitle = ""

    def __init__(self, master, app):
        super().__init__(master, bg=BG)
        self.app = app

        # ── scrollable interior ───────────────────────────────────────────
        # Every step gets a Canvas + inner Frame so content can scroll when
        # the window is shorter than the content.
        self._canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        self._vsb = tk.Scrollbar(self, orient="vertical", command=self._canvas.yview,
                                 bg=BG2, troughcolor=BG3, relief="flat")
        self._canvas.configure(yscrollcommand=self._vsb.set)

        self._vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        # Inner frame that actually holds all widgets
        self.interior = tk.Frame(self._canvas, bg=BG)
        self._win_id = self._canvas.create_window((0, 0), window=self.interior,
                                                   anchor="nw")

        # Resize events keep the inner frame matching the canvas width
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self.interior.bind("<Configure>", self._on_interior_configure)

        # Mousewheel scrolling (Windows + macOS/Linux)
        self._canvas.bind("<Enter>", self._bind_mousewheel)
        self._canvas.bind("<Leave>", self._unbind_mousewheel)

    # --- keep inner frame at least as wide as the canvas ---
    def _on_canvas_configure(self, event):
        self._canvas.itemconfig(self._win_id, width=event.width)

    def _on_interior_configure(self, _event):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _bind_mousewheel(self, _event):
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)          # Windows
        self._canvas.bind_all("<Button-4>", self._on_mousewheel_linux)      # Linux up
        self._canvas.bind_all("<Button-5>", self._on_mousewheel_linux)      # Linux down

    def _unbind_mousewheel(self, _event):
        self._canvas.unbind_all("<MouseWheel>")
        self._canvas.unbind_all("<Button-4>")
        self._canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self._canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._canvas.yview_scroll(1, "units")

    def on_enter(self):
        """Called when the wizard navigates TO this step."""
        pass

    def validate(self) -> bool:
        return True


# ─── Step 1: Select Model ─────────────────────────────────────────────────────

class StepModel(StepBase):
    title    = "Step 1 — Select Model"
    subtitle = "Choose model session and checkpoint"

    def __init__(self, master, app):
        super().__init__(master, app)
        self._ckpt_var    = self.app.vars["checkpoint"]
        self._custom_var  = self.app.vars["ckpt_custom"]
        self._session_var = self.app.vars["model_session"]
        self._sessions    = []
        self._build()

    def _build(self):
        # ── Model session selector ────────────────────────────────────────
        section_label(self.interior, "Model Session")
        sc = card(self.interior)
        si = card_inner(sc)
        sess_row = tk.Frame(si, bg=BG2)
        sess_row.pack(fill="x")
        lbl(sess_row, "Session:", bold=True).pack(side="left", padx=(0, 8))
        self._sess_combo = dark_combo(sess_row, self._session_var, [], width=22)
        self._sess_combo.pack(side="left")
        self._sess_combo.bind("<<ComboboxSelected>>", lambda _: self._on_session_change())
        styled_button(sess_row, "↻ Scan", self._scan_sessions, padx=8, pady=3).pack(
            side="left", padx=8)
        self._sess_type_lbl = lbl(si, "", dim=True)
        self._sess_type_lbl.pack(anchor="w", pady=(4, 0))

        # ── Checkpoint selection ──────────────────────────────────────────
        section_label(self.interior, "Checkpoint")
        self._best_info_lbl = None
        self._last_info_lbl = None

        for val, title in [
            ("best", "Best checkpoint  (best.pt)"),
            ("last", "Last checkpoint  (last.pt)"),
            ("custom", "Custom path …"),
        ]:
            c = card(self.interior)
            inner = card_inner(c)
            rb = tk.Radiobutton(
                inner, text=title, variable=self._ckpt_var, value=val,
                font=FONT_B, bg=BG2, fg=FG, selectcolor=BG,
                activebackground=BG2, activeforeground=ACCENT,
                command=self._on_ckpt_change,
            )
            rb.pack(anchor="w")
            info_lbl = tk.Label(inner, text="", font=FONT_MONO, bg=BG2, fg=FG_DIM,
                                anchor="w")
            info_lbl.pack(anchor="w", padx=20)
            if val == "best":
                self._best_info_lbl = info_lbl
            elif val == "last":
                self._last_info_lbl = info_lbl
            elif val == "custom":
                info_lbl.config(text="Browse to any .pt file")

        # Custom path row
        self._custom_frame = tk.Frame(self.interior, bg=BG)
        self._custom_frame.pack(fill="x", pady=4)
        cf = tk.Frame(self._custom_frame, bg=BG2)
        cf.pack(fill="x", padx=0, pady=2)
        self._custom_entry = entry_field(cf, self._custom_var, width=40, state="disabled")
        self._custom_entry.pack(side="left", fill="x", expand=True, padx=(10, 4), pady=6)
        self._browse_btn = styled_button(cf, "Browse …", self._browse, padx=8, pady=4)
        self._browse_btn.config(state="disabled")
        self._browse_btn.pack(side="left", padx=(0, 10))

        section_label(self.interior, "No-Model Mode")
        nm = card(self.interior)
        ni = card_inner(nm)
        tk.Checkbutton(
            ni, text="Skip model inference  (compare clean vs noisy only)",
            variable=self.app.vars["no_model"],
            font=FONT, bg=BG2, fg=FG, selectcolor=BG,
            activebackground=BG2, activeforeground=ACCENT,
        ).pack(anchor="w")

        self._on_ckpt_change()
        self._scan_sessions()

    def _scan_sessions(self):
        """Scan outputs/ for all model session directories."""
        self._sessions = _scan_model_sessions()
        names = [s["name"] for s in self._sessions]
        self._sess_combo["values"] = names
        # Auto-select first session with a checkpoint
        current = self._session_var.get()
        if current not in names and names:
            self._session_var.set(names[-1])
        self._on_session_change()

    def _on_session_change(self):
        """Update checkpoint info when session changes."""
        name = self._session_var.get()
        sess = next((s for s in self._sessions if s["name"] == name), None)
        if sess is None:
            self._sess_type_lbl.config(text="No session selected")
            if self._best_info_lbl:
                self._best_info_lbl.config(text="not found")
            if self._last_info_lbl:
                self._last_info_lbl.config(text="not found")
            return

        model_type = sess["model_type"].upper()
        self._sess_type_lbl.config(
            text=f"Model type: {model_type}   |   Path: {sess['path']}")

        best_path = sess["path"] / "best.pt"
        last_path = sess["path"] / "last.pt"
        if self._best_info_lbl:
            self._best_info_lbl.config(text=_auto_ckpt_info(best_path))
        if self._last_info_lbl:
            self._last_info_lbl.config(text=_auto_ckpt_info(last_path))

    def _get_ckpt_dir(self) -> Path:
        """Get checkpoint directory for the currently selected session."""
        name = self._session_var.get()
        sess = next((s for s in self._sessions if s["name"] == name), None)
        if sess is not None:
            return sess["path"]
        return config.CHECKPOINT_DIR

    def _on_ckpt_change(self):
        custom = self._ckpt_var.get() == "custom"
        state  = "normal" if custom else "disabled"
        self._custom_entry.config(state=state)
        self._browse_btn.config(state=state)

    def _browse(self):
        f = filedialog.askopenfilename(
            title="Select checkpoint (.pt)",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
            initialdir=str(self._get_ckpt_dir()),
        )
        if f:
            self._custom_var.set(f)

    def validate(self):
        ckpt = self._ckpt_var.get()
        if self.app.vars["no_model"].get():
            return True
        ckpt_dir = self._get_ckpt_dir()
        if ckpt == "best":
            p = ckpt_dir / "best.pt"
        elif ckpt == "last":
            p = ckpt_dir / "last.pt"
        else:
            p = Path(self._custom_var.get().strip())
        if not p.exists():
            messagebox.showerror("Missing checkpoint",
                                 f"Checkpoint not found:\n{p}\n\nTrain the model first.",
                                 parent=self.app.root)
            return False
        return True


# ─── Step 2: Data Source ──────────────────────────────────────────────────────

class StepDataSource(StepBase):
    title    = "Step 2 — Data Source"
    subtitle = "Where should the ECG signal come from?"

    def __init__(self, master, app):
        super().__init__(master, app)
        self._mode     = self.app.vars["mode"]
        self._csv_dir  = self.app.vars["csv_dir"]
        self._raw_dir  = self.app.vars["raw_dir"]
        self._record   = self.app.vars["record"]
        self._build()

    def _build(self):
        for val, title, sub in [
            ("A", "Mode A — Prepared DATA/ folder",
             "Use .npy files already created by prepare_data.py"),
            ("B", "Mode B — CSV outputs/ folder",
             "Window CSV files on-the-fly"),
            ("C", "Mode C — Raw WFDB record",
             "Convert a raw MIT-BIH .dat/.hea record"),
        ]:
            c = card(self.interior)
            inner = card_inner(c)
            rb = tk.Radiobutton(
                inner, text=title, variable=self._mode, value=val,
                font=FONT_B, bg=BG2, fg=FG, selectcolor=BG,
                activebackground=BG2, activeforeground=ACCENT,
                command=self._on_mode_change,
            )
            rb.pack(anchor="w")
            tk.Label(inner, text=sub, font=FONT, bg=BG2, fg=FG_DIM,
                     anchor="w").pack(anchor="w", padx=20)

        # Mode B CSV dir
        section_label(self.interior, "Mode B — CSV Directory")
        self._b_frame = tk.Frame(self.interior, bg=BG)
        self._b_frame.pack(fill="x")
        bf = tk.Frame(self._b_frame, bg=BG2)
        bf.pack(fill="x", padx=0, pady=2)
        self._csv_entry = entry_field(bf, self._csv_dir, width=40)
        self._csv_entry.pack(side="left", fill="x", expand=True, padx=(10, 4), pady=6)
        styled_button(bf, "Browse …", self._browse_csv, padx=8, pady=4).pack(
            side="left", padx=(0, 10))

        # Mode C WFDB dir + record picker
        section_label(self.interior, "Mode C — WFDB Directory & Record")
        self._c_frame = tk.Frame(self.interior, bg=BG)
        self._c_frame.pack(fill="x")

        # Dir row
        cf = tk.Frame(self._c_frame, bg=BG2)
        cf.pack(fill="x", padx=0, pady=2)
        self._raw_entry = entry_field(cf, self._raw_dir, width=36)
        self._raw_entry.pack(side="left", fill="x", expand=True, padx=(10, 4), pady=6)
        styled_button(cf, "Browse …", self._browse_raw, padx=8, pady=4).pack(
            side="left", padx=(0, 10))
        self._raw_dir.trace_add("write", lambda *_: self._refresh_wfdb_records())

        # Record picker label
        rf_lbl = tk.Frame(self._c_frame, bg=BG2)
        rf_lbl.pack(fill="x", padx=0, pady=(4, 0))
        lbl(rf_lbl, "Available records (click to select):").pack(
            side="left", padx=(10, 0), pady=2)
        self._wfdb_count_lbl = lbl(rf_lbl, "", dim=True)
        self._wfdb_count_lbl.pack(side="left", padx=8)

        # Scrollable record listbox
        lb_outer = tk.Frame(self._c_frame, bg=BG2)
        lb_outer.pack(fill="x", padx=0, pady=2)
        lb_inner = tk.Frame(lb_outer, bg=BG2)
        lb_inner.pack(fill="x", padx=10, pady=4)
        sb = tk.Scrollbar(lb_inner, bg=BG2, troughcolor=BG3, relief="flat")
        sb.pack(side="right", fill="y")
        self._wfdb_lb = tk.Listbox(
            lb_inner, font=FONT_MONO, bg=BG3, fg=FG,
            selectbackground=ACCENT, selectforeground=BG,
            activestyle="none", relief="flat",
            yscrollcommand=sb.set, height=5, exportselection=False,
        )
        self._wfdb_lb.pack(side="left", fill="x", expand=True)
        sb.config(command=self._wfdb_lb.yview)
        self._wfdb_lb.bind("<<ListboxSelect>>", self._on_wfdb_select)

        # Selected record display
        sel_row = tk.Frame(self._c_frame, bg=BG2)
        sel_row.pack(fill="x", padx=0, pady=(2, 6))
        lbl(sel_row, "Selected record:", bold=True).pack(side="left", padx=(10, 6))
        self._selected_lbl = tk.Label(sel_row, textvariable=self._record,
                                      font=FONT_B, bg=BG2, fg=ACCENT)
        self._selected_lbl.pack(side="left")

        self._on_mode_change()

    def _on_mode_change(self):
        m = self._mode.get()
        # show/hide optional frames
        if m == "B":
            self._b_frame.pack(fill="x")
        else:
            self._b_frame.pack_forget()
        if m == "C":
            self._c_frame.pack(fill="x")
        else:
            self._c_frame.pack_forget()

    def _browse_csv(self):
        d = filedialog.askdirectory(title="Select CSV directory",
                                    initialdir=str(config.DATA_DIR))
        if d:
            self._csv_dir.set(d)

    def _browse_raw(self):
        d = filedialog.askdirectory(title="Select WFDB directory",
                                    initialdir=str(config.BASE_DIR))
        if d:
            self._raw_dir.set(d)
            self._refresh_wfdb_records()

    def _refresh_wfdb_records(self):
        """Scan raw_dir for .hea files and populate the record listbox."""
        raw_dir = Path(self._raw_dir.get().strip())
        self._wfdb_lb.delete(0, "end")
        if not raw_dir.exists():
            self._wfdb_count_lbl.config(text="(directory not found)")
            return
        # Collect unique record stems that have both .hea and .dat
        records = sorted({
            p.stem for p in raw_dir.glob("*.hea")
            if (raw_dir / (p.stem + ".dat")).exists()
            and not p.stem.endswith("-0")   # skip alternate annotation files
        })
        for r in records:
            self._wfdb_lb.insert("end", f"  {r}")
        count = len(records)
        self._wfdb_count_lbl.config(text=f"({count} record{'s' if count != 1 else ''} found)")
        # Auto-select first if nothing chosen yet
        if records and not self._record.get():
            self._wfdb_lb.selection_set(0)
            self._record.set(records[0])

    def _on_wfdb_select(self, _event):
        sel = self._wfdb_lb.curselection()
        if sel:
            val = self._wfdb_lb.get(sel[0]).strip()
            self._record.set(val)

    def validate(self):
        m = self._mode.get()
        if m == "B" and not self._csv_dir.get().strip():
            messagebox.showerror("Missing", "Please select a CSV directory.",
                                 parent=self.app.root)
            return False
        if m == "C":
            if not self._raw_dir.get().strip():
                messagebox.showerror("Missing", "Please select a WFDB directory.",
                                     parent=self.app.root)
                return False
            if not self._record.get().strip():
                messagebox.showerror("Missing",
                                     "Please select a record from the list.",
                                     parent=self.app.root)
                return False
        return True

    def on_enter(self):
        self._on_mode_change()
        if self._mode.get() == "C" and self._raw_dir.get().strip():
            self._refresh_wfdb_records()


# ─── Step 3: Select Patient ───────────────────────────────────────────────────

class StepPatient(StepBase):
    title    = "Step 3 — Select Patient & Split"
    subtitle = "Choose the patient and data split to test on"

    def __init__(self, master, app):
        super().__init__(master, app)
        self._split   = self.app.vars["split"]
        self._patient = self.app.vars["patient"]
        self._build()

    def _build(self):
        section_label(self.interior, "Data Split")
        sc = card(self.interior)
        si = card_inner(sc)
        split_row = tk.Frame(si, bg=BG2)
        split_row.pack(fill="x")
        for val in ("train", "val", "test"):
            tk.Radiobutton(
                split_row, text=val.capitalize(),
                variable=self._split, value=val,
                font=FONT, bg=BG2, fg=FG, selectcolor=BG,
                activebackground=BG2, activeforeground=ACCENT,
                command=self._refresh,
            ).pack(side="left", padx=12)

        section_label(self.interior, "Patient")
        pc = card(self.interior)
        pi = card_inner(pc)

        pr = tk.Frame(pi, bg=BG2)
        pr.pack(fill="x")
        lbl(pr, "Patient ID:", bold=True).pack(side="left", padx=(0, 8))
        self._combo = dark_combo(pr, self._patient, [], width=18)
        self._combo.pack(side="left")
        styled_button(pr, "↻ Refresh", self._refresh, padx=8, pady=3).pack(
            side="left", padx=8)

        lbl(pi, "(auto-populated from DATA/ for Mode A)", dim=True).pack(
            anchor="w", pady=4)

        # Found patients list
        self._list_frame = tk.Frame(self.interior, bg=BG)
        self._list_frame.pack(fill="both", expand=True, pady=8)
        section_label(self._list_frame, "Available Patients (Mode A)")
        self._listbox_var = tk.StringVar()
        lb_frame = tk.Frame(self._list_frame, bg=BG2)
        lb_frame.pack(fill="both", expand=True)
        sb = tk.Scrollbar(lb_frame, bg=BG2, troughcolor=BG3, relief="flat")
        sb.pack(side="right", fill="y")
        self._listbox = tk.Listbox(
            lb_frame, font=FONT_MONO, bg=BG3, fg=FG, selectbackground=ACCENT,
            selectforeground=BG, activestyle="none", relief="flat",
            yscrollcommand=sb.set, height=6,
        )
        self._listbox.pack(side="left", fill="both", expand=True)
        sb.config(command=self._listbox.yview)
        self._listbox.bind("<<ListboxSelect>>", self._on_lb_select)

    def _refresh(self):
        split   = self._split.get()
        mode    = self.app.vars["mode"].get()
        current = self._patient.get()

        if mode == "A":
            patients = _auto_patients(split)
        elif mode == "B":
            patients = _auto_csv_patients(self.app.vars["csv_dir"].get(), split)
        else:
            patients = []

        self._combo["values"] = patients
        self._listbox.delete(0, "end")
        for p in patients:
            self._listbox.insert("end", f"  {p}")

        if patients:
            if current not in patients:
                self._patient.set(patients[0])
        else:
            self._patient.set("")

    def _on_lb_select(self, _):
        sel = self._listbox.curselection()
        if sel:
            val = self._listbox.get(sel[0]).strip()
            self._patient.set(val)
            self._combo.set(val)

    def on_enter(self):
        self._refresh()

    def validate(self):
        mode = self.app.vars["mode"].get()
        if mode == "C":
            return True  # no patient needed for WFDB
        if not self._patient.get().strip():
            messagebox.showerror("Missing", "Please select or enter a patient ID.",
                                 parent=self.app.root)
            return False
        return True


# ─── Step 4: Configure Test ───────────────────────────────────────────────────

class StepConfigure(StepBase):
    title    = "Step 4 — Configure Test"
    subtitle = "Set SNR, batch size, and display options"

    def __init__(self, master, app):
        super().__init__(master, app)
        self._snr        = self.app.vars["snr"]
        self._n_windows  = self.app.vars["n_windows"]
        self._batch      = self.app.vars["batch_size"]
        self._max_win    = self.app.vars["max_windows"]
        self._build()

    def _build(self):
        section_label(self.interior, "Input Noise Level (SNR)")
        nc = card(self.interior)
        ni = card_inner(nc)
        snr_row = tk.Frame(ni, bg=BG2)
        snr_row.pack(fill="x")
        presets = [
            (f"Phase 4 HIGH  ({config.SNR_HIGH} dB)", config.SNR_HIGH),
            (f"Phase 3 MID   ({config.SNR_MEDIUM} dB)", config.SNR_MEDIUM),
            (f"Phase 2 LOW   ({config.SNR_LOW} dB)",  config.SNR_LOW),
            ("Custom", None),
        ]
        self._snr_preset = tk.StringVar(value=f"Phase 4 HIGH  ({config.SNR_HIGH} dB)")
        for txt, val in presets:
            rb = tk.Radiobutton(
                snr_row, text=txt,
                variable=self._snr_preset, value=txt,
                font=FONT, bg=BG2, fg=FG, selectcolor=BG,
                activebackground=BG2, activeforeground=ACCENT,
                command=lambda v=val: self._set_snr(v),
            )
            rb.pack(anchor="w", pady=1)

        custom_row = tk.Frame(ni, bg=BG2)
        custom_row.pack(fill="x", pady=4)
        lbl(custom_row, "Custom SNR (dB):", bold=True).pack(side="left", padx=(0, 8))
        self._snr_scale = tk.Scale(
            custom_row, from_=-5, to=40,
            variable=self._snr, orient="horizontal",
            resolution=0.5, bg=BG2, fg=FG,
            troughcolor=ENTRY_BG, highlightbackground=BG2,
            activebackground=ACCENT, sliderrelief="flat",
            font=("Segoe UI", 7), length=200,
            command=lambda _: self._snr_preset.set("Custom"),
        )
        self._snr_scale.pack(side="left")
        tk.Label(custom_row, textvariable=self._snr,
                 font=FONT_B, bg=BG2, fg=ACCENT, width=6).pack(side="left", padx=6)

        section_label(self.interior, "Display Options")
        dc = card(self.interior)
        di = card_inner(dc)
        dr = tk.Frame(di, bg=BG2)
        dr.pack(fill="x", pady=2)
        lbl(dr, "Overview windows:", bold=True).pack(side="left", padx=(0, 8))
        tk.Spinbox(
            dr, from_=1, to=100, textvariable=self._n_windows,
            width=5, bg=ENTRY_BG, fg=FG, insertbackground=FG,
            buttonbackground=BG2, relief="flat", font=FONT,
        ).pack(side="left")
        lbl(dr, "windows shown in strip view", dim=True).pack(side="left", padx=8)

        br = tk.Frame(di, bg=BG2)
        br.pack(fill="x", pady=6)
        lbl(br, "Inference batch size:", bold=True).pack(side="left", padx=(0, 8))
        tk.Spinbox(
            br, from_=1, to=512, textvariable=self._batch,
            width=6, bg=ENTRY_BG, fg=FG, insertbackground=FG,
            buttonbackground=BG2, relief="flat", font=FONT,
        ).pack(side="left")
        lbl(br, "(reduce if you get OOM errors)", dim=True).pack(side="left", padx=8)

        mr = tk.Frame(di, bg=BG2)
        mr.pack(fill="x", pady=6)
        lbl(mr, "Max windows:", bold=True).pack(side="left", padx=(0, 8))
        tk.Spinbox(
            mr, from_=50, to=20000, increment=50, textvariable=self._max_win,
            width=7, bg=ENTRY_BG, fg=FG, insertbackground=FG,
            buttonbackground=BG2, relief="flat", font=FONT,
        ).pack(side="left")
        lbl(mr, "(cap raw recordings to avoid OOM — 2000 recommended)", dim=True
            ).pack(side="left", padx=8)

        # ── Lead display selection ────────────────────────────────────────
        section_label(self.interior, "Lead Display")
        lc = card(self.interior)
        li = card_inner(lc)
        lead_row = tk.Frame(li, bg=BG2)
        lead_row.pack(fill="x")
        lbl(lead_row, "Show leads:", bold=True).pack(side="left", padx=(0, 12))
        lead_var = self.app.vars["lead_display"]
        for val, txt in [("both", "Both Leads"), ("1", "Lead I only"), ("2", "Lead II only")]:
            tk.Radiobutton(
                lead_row, text=txt, variable=lead_var, value=val,
                font=FONT, bg=BG2, fg=FG, selectcolor=BG,
                activebackground=BG2, activeforeground=ACCENT,
            ).pack(side="left", padx=8)

        # ── Performance / CPU core limiting ───────────────────────────────
        section_label(self.interior, "Performance")
        perf_c = card(self.interior)
        perf_i = card_inner(perf_c)
        cpu_row = tk.Frame(perf_i, bg=BG2)
        cpu_row.pack(fill="x", pady=2)
        lbl(cpu_row, "CPU core limit:", bold=True).pack(side="left", padx=(0, 8))
        tk.Spinbox(
            cpu_row, from_=1, to=os.cpu_count() or 16,
            textvariable=self.app.vars["cpu_cores"],
            width=4, bg=ENTRY_BG, fg=FG, insertbackground=FG,
            buttonbackground=BG2, relief="flat", font=FONT,
        ).pack(side="left")
        lbl(cpu_row, f"(max {os.cpu_count() or '?'} available — restricts PyTorch threads)",
            dim=True).pack(side="left", padx=8)

        # ── Data percentage limit ─────────────────────────────────────────
        pct_row = tk.Frame(perf_i, bg=BG2)
        pct_row.pack(fill="x", pady=6)
        lbl(pct_row, "Data % per patient:", bold=True).pack(side="left", padx=(0, 8))
        tk.Spinbox(
            pct_row, from_=5, to=100, increment=5,
            textvariable=self.app.vars["data_percent"],
            width=5, bg=ENTRY_BG, fg=FG, insertbackground=FG,
            buttonbackground=BG2, relief="flat", font=FONT,
        ).pack(side="left")
        lbl(pct_row, "% of each patient's windows (faster testing with 10-20%)",
            dim=True).pack(side="left", padx=8)

        # ── Random Benchmark ──────────────────────────────────────────────
        section_label(self.interior, "Random Multi-Patient Benchmark")
        bench_c = card(self.interior)
        bench_i = card_inner(bench_c)
        bench_desc = tk.Frame(bench_i, bg=BG2)
        bench_desc.pack(fill="x")
        lbl(bench_desc,
            "Pick random patients, noise their data, run model, and calculate all metrics.",
            dim=True).pack(anchor="w")

        bench_row = tk.Frame(bench_i, bg=BG2)
        bench_row.pack(fill="x", pady=6)
        lbl(bench_row, "Number of patients:", bold=True).pack(side="left", padx=(0, 8))
        tk.Spinbox(
            bench_row, from_=1, to=20,
            textvariable=self.app.vars["bench_n_patients"],
            width=4, bg=ENTRY_BG, fg=FG, insertbackground=FG,
            buttonbackground=BG2, relief="flat", font=FONT,
        ).pack(side="left")
        lbl(bench_row, "random patients to test", dim=True).pack(side="left", padx=8)

        self._bench_btn = styled_button(
            bench_i, "Run Benchmark", self._run_benchmark, accent=True,
            padx=16, pady=8)
        self._bench_btn.pack(anchor="w", pady=(8, 2))
        self._bench_status = tk.Label(bench_i, text="", font=FONT_MONO,
                                      bg=BG2, fg=FG_DIM, anchor="w")
        self._bench_status.pack(anchor="w")

        section_label(self.interior, "Summary")
        self._summary_frame = tk.Frame(self.interior, bg=BG2)
        self._summary_frame.pack(fill="x", pady=2)
        self._summary_lbl = tk.Label(
            self._summary_frame, text="", font=FONT_MONO,
            bg=BG2, fg=FG_DIM, justify="left", anchor="w",
        )
        self._summary_lbl.pack(padx=14, pady=8, anchor="w")

    def _set_snr(self, val):
        if val is not None:
            self._snr.set(float(val))

    def _run_benchmark(self):
        """Launch the random multi-patient benchmark from StepConfigure."""
        self._bench_btn.config(state="disabled", text="Running ...")
        self._bench_status.config(text="Starting benchmark ...", fg=ACCENT)
        self.app.after(100, lambda: self.app.run_benchmark(self._bench_status, self._bench_btn))

    def on_enter(self):
        v = self.app.vars
        lines = [
            f"  Model session:  {v['model_session'].get()}",
            f"  Checkpoint  :  {v['checkpoint'].get()}",
            f"  Mode        :  {v['mode'].get()}",
            f"  Patient     :  {v['patient'].get() or v['record'].get()}",
            f"  Split       :  {v['split'].get()}",
            f"  No model    :  {v['no_model'].get()}",
            f"  Lead display:  {v['lead_display'].get()}",
            f"  CPU cores   :  {v['cpu_cores'].get()}",
            f"  Data %      :  {v['data_percent'].get()}%",
        ]
        self._summary_lbl.config(text="\n".join(lines))


# ─── Step 5: Processing ───────────────────────────────────────────────────────

class StepProcessing(StepBase):
    title    = "Step 5 — Processing"
    subtitle = "Loading data and running inference …"

    def __init__(self, master, app):
        super().__init__(master, app)
        self._last_log_lines = []
        self._build()

    def _build(self):
        section_label(self.interior, "Progress")

        # Progress bar
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Horizontal.TProgressbar",
                        troughcolor=BG2, background=ACCENT,
                        bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT)
        pc = card(self.interior)
        pi = card_inner(pc)
        self._prog = ttk.Progressbar(pi, style="Dark.Horizontal.TProgressbar",
                                     mode="indeterminate", length=400)
        self._prog.pack(fill="x", pady=4)
        self._status = tk.Label(pi, text="Starting …", font=FONT_B,
                                bg=BG2, fg=ACCENT, anchor="w")
        self._status.pack(anchor="w")

        section_label(self.interior, "Log")
        lc = tk.Frame(self.interior, bg=BG2)
        lc.pack(fill="both", expand=True, pady=4)
        self._log_box = scrolledtext.ScrolledText(
            lc, font=FONT_MONO, bg=BG3, fg=FG_DIM,
            relief="flat", state="disabled", height=14,
            wrap="word",
        )
        self._log_box.pack(fill="both", expand=True, padx=4, pady=4)

    def log(self, msg: str, replace_last: bool = False):
        """Thread-safe log append."""
        self._log_box.after(0, self._append_log, msg, replace_last)

    def _append_log(self, msg: str, replace_last: bool):
        self._log_box.config(state="normal")
        if replace_last and self._last_log_lines:
            # Remove last line
            self._log_box.delete("end-2l", "end-1l")
        self._log_box.insert("end", msg + "\n")
        self._log_box.see("end")
        self._log_box.config(state="disabled")
        if replace_last and self._last_log_lines:
            self._last_log_lines[-1] = msg
        else:
            self._last_log_lines.append(msg)

    def set_status(self, msg: str):
        self._status.after(0, self._status.config, {"text": msg})

    def start_spinner(self):
        self._prog.start(12)

    def stop_spinner(self):
        self._prog.stop()
        self._prog.config(mode="determinate", value=100)

    def on_enter(self):
        # Clear log
        self._log_box.config(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.config(state="disabled")
        self._last_log_lines.clear()
        self._prog.config(mode="indeterminate", value=0)
        self._status.config(text="Starting …")
        # Kick off background thread
        self.app.after(100, self.app.run_processing)


# ─── Step 6: Results Viewer ───────────────────────────────────────────────────

class StepResults(StepBase):
    title    = "Step 6 — Results"
    subtitle = "Interactive ECG viewer"

    def __init__(self, master, app):
        super().__init__(master, app)
        self._canvas_widget = None
        self._fig           = None
        self._vis           = None
        self._build_frame()

    def _build_frame(self):
        self._plot_container = tk.Frame(self.interior, bg=BG)
        self._plot_container.pack(fill="both", expand=True)

        self._metrics_bar = tk.Label(
            self.interior, text="", font=FONT_MONO, bg=BG2, fg=ACCENT,
            anchor="w", justify="left",
        )
        self._metrics_bar.pack(fill="x", padx=6, pady=4)

    def load_results(self, results: dict, metrics: dict, patient: str, snr_db: float,
                     model=None, device=None, lead_display="both", model_type=None):
        """Called after processing completes."""
        if self._canvas_widget:
            self._canvas_widget.get_tk_widget().destroy()
            plt.close(self._fig)

        self._vis = ECGViewer(
            results=results,
            metrics=metrics,
            patient=patient,
            snr_db=snr_db,
            n_windows=self.app.vars["n_windows"].get(),
            container=self._plot_container,
            model=model,
            device=device,
            lead_display=lead_display,
            model_type=model_type,
        )
        self._canvas_widget = self._vis.canvas

        m = metrics
        def _fmt(v, fmt=".2f"):
            if isinstance(v, float) and v != v:
                return "N/A"
            return format(v, fmt)
        bar_txt = (
            f"  Patient: {patient}   SNR: {snr_db} dB  ▸  "
            f"SNR  {_fmt(m['snr_noisy'], '+.2f')} dB → {_fmt(m['snr_denoised'], '+.2f')} dB  "
            f"(Δ {_fmt(m['snr_improve'], '+.2f')} dB)    "
            f"PRD  {_fmt(m['prd_noisy'], '.2f')}% → {_fmt(m['prd_denoised'], '.2f')}%    "
            f"MSE  {_fmt(m['mse_noisy'], '.5f')} → {_fmt(m['mse_denoised'], '.5f')}"
        )
        self._metrics_bar.config(text=bar_txt)


# ═════════════════════════════════════════════════════════════════════════════
# Benchmark Results Window (Toplevel)
# ═════════════════════════════════════════════════════════════════════════════

class BenchmarkResultsWindow(tk.Toplevel):
    """
    Special results window showing per-patient and aggregate benchmark metrics
    with explanations of each score.
    """

    METRIC_EXPLANATIONS = {
        "SNR (Input)": (
            "Signal-to-Noise Ratio of the noisy input.\n"
            "Measures how much noise was added. Lower values = more noise.\n"
            "Formula: 10 * log10(signal_power / noise_power)\n"
            "Unit: dB (decibels). Typical noisy ECG: 0-12 dB."
        ),
        "SNR (Output)": (
            "Signal-to-Noise Ratio after denoising.\n"
            "Higher is better — means the model removed noise effectively.\n"
            "Good: > 20 dB, Excellent: > 25 dB."
        ),
        "Delta SNR": (
            "Improvement in SNR: SNR(output) - SNR(input).\n"
            "Shows how many dB the model improved the signal.\n"
            "Positive = improvement. Higher = better denoising.\n"
            "Good: > 5 dB, Excellent: > 10 dB."
        ),
        "PRD (Input)": (
            "Percent Root-mean-square Difference of noisy input.\n"
            "Measures distortion relative to clean signal.\n"
            "Formula: 100 * sqrt(sum((pred-clean)^2) / sum(clean^2))\n"
            "Unit: %. Lower is better for denoised output."
        ),
        "PRD (Output)": (
            "Percent Root-mean-square Difference after denoising.\n"
            "Lower is better — means less distortion from the clean signal.\n"
            "Good: < 10%, Excellent: < 5%."
        ),
        "MSE (Input)": (
            "Mean Squared Error between noisy and clean signal.\n"
            "Average squared difference per sample.\n"
            "Lower is better. Scale depends on signal amplitude."
        ),
        "MSE (Output)": (
            "Mean Squared Error between denoised and clean signal.\n"
            "Lower is better — means the denoised signal is closer to clean.\n"
            "Should be significantly lower than MSE(Input)."
        ),
    }

    def __init__(self, parent, bench_results: list, aggregate: dict,
                 snr_db: float, cpu_cores: int, data_pct: int,
                 elapsed: float, model_type: str):
        super().__init__(parent)
        self.title("Benchmark Results")
        self.configure(bg=BG)
        self.minsize(900, 600)
        self.transient(parent)

        self._build(bench_results, aggregate, snr_db, cpu_cores, data_pct,
                    elapsed, model_type)
        self._center()

    def _center(self):
        self.update_idletasks()
        w, h = 1000, 700
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _build(self, bench_results, aggregate, snr_db, cpu_cores, data_pct,
               elapsed, model_type):
        # Header
        tk.Frame(self, bg=ACCENT, height=3).pack(fill="x")
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(10, 4))
        tk.Label(hdr, text="Benchmark Results",
                 font=("Segoe UI", 14, "bold"), bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(hdr, text=f"  |  {model_type.upper()} model  |  "
                           f"{len(bench_results)} patients  |  "
                           f"SNR={snr_db} dB  |  "
                           f"{data_pct}% data  |  "
                           f"{cpu_cores} CPU cores  |  "
                           f"{elapsed:.1f}s",
                 font=FONT, bg=BG, fg=FG_DIM).pack(side="left", padx=8)

        tk.Frame(self, bg=SEP, height=1).pack(fill="x", pady=4)

        # Main scrollable area
        main_frame = tk.Frame(self, bg=BG)
        main_frame.pack(fill="both", expand=True, padx=20, pady=4)

        canvas = tk.Canvas(main_frame, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=BG)
        scroll_frame.bind("<Configure>",
                         lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Bind mouse wheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.protocol("WM_DELETE_WINDOW", lambda: (canvas.unbind_all("<MouseWheel>"),
                                                    self.destroy()))

        # ── Per-patient results table ─────────────────────────────────────
        section_label(scroll_frame, "Per-Patient Results")
        table_frame = tk.Frame(scroll_frame, bg=BG2)
        table_frame.pack(fill="x", pady=4)

        # Table headers
        headers = ["Patient", "Windows", "SNR In", "SNR Out", "Delta SNR",
                   "PRD In", "PRD Out", "MSE In", "MSE Out"]
        for col, h in enumerate(headers):
            tk.Label(table_frame, text=h, font=FONT_B, bg=BG3, fg=ACCENT,
                     padx=10, pady=6, anchor="center"
                     ).grid(row=0, column=col, sticky="nsew", padx=1, pady=1)

        # Table rows
        for row_i, r in enumerate(bench_results, start=1):
            m = r["metrics"]
            vals = [
                r["patient"],
                str(r["n_windows"]),
                f"{m['snr_noisy']:+.2f}",
                f"{m['snr_denoised']:+.2f}" if not _is_nan(m['snr_denoised']) else "N/A",
                f"{m['snr_improve']:+.2f}" if not _is_nan(m['snr_improve']) else "N/A",
                f"{m['prd_noisy']:.2f}",
                f"{m['prd_denoised']:.2f}" if not _is_nan(m['prd_denoised']) else "N/A",
                f"{m['mse_noisy']:.6f}",
                f"{m['mse_denoised']:.6f}" if not _is_nan(m['mse_denoised']) else "N/A",
            ]
            bg_row = BG2 if row_i % 2 == 0 else BG3
            for col, v in enumerate(vals):
                fg_c = FG
                if col == 4 and v != "N/A":  # Delta SNR coloring
                    fg_c = GREEN if float(v) > 0 else RED
                tk.Label(table_frame, text=v, font=FONT_MONO, bg=bg_row,
                         fg=fg_c, padx=10, pady=4, anchor="center"
                         ).grid(row=row_i, column=col, sticky="nsew", padx=1, pady=1)

        for col in range(len(headers)):
            table_frame.columnconfigure(col, weight=1)

        # ── Aggregate results ─────────────────────────────────────────────
        section_label(scroll_frame, "Aggregate Results (Average)")
        agg_frame = tk.Frame(scroll_frame, bg=BG2)
        agg_frame.pack(fill="x", pady=4)
        agg_inner = tk.Frame(agg_frame, bg=BG2)
        agg_inner.pack(fill="x", padx=14, pady=10)

        agg_items = [
            ("SNR (Input)",  f"{aggregate['snr_noisy']:+.2f} dB"),
            ("SNR (Output)", f"{aggregate['snr_denoised']:+.2f} dB"
                             if not _is_nan(aggregate['snr_denoised']) else "N/A"),
            ("Delta SNR",    f"{aggregate['snr_improve']:+.2f} dB"
                             if not _is_nan(aggregate['snr_improve']) else "N/A"),
            ("PRD (Input)",  f"{aggregate['prd_noisy']:.2f} %"),
            ("PRD (Output)", f"{aggregate['prd_denoised']:.2f} %"
                             if not _is_nan(aggregate['prd_denoised']) else "N/A"),
            ("MSE (Input)",  f"{aggregate['mse_noisy']:.6f}"),
            ("MSE (Output)", f"{aggregate['mse_denoised']:.6f}"
                             if not _is_nan(aggregate['mse_denoised']) else "N/A"),
        ]

        for i, (label, value) in enumerate(agg_items):
            row = tk.Frame(agg_inner, bg=BG2)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"  {label}:", font=FONT_B, bg=BG2, fg=FG,
                     width=16, anchor="w").pack(side="left")
            fg_c = FG
            if "Delta" in label and value != "N/A":
                fg_c = GREEN if float(value.replace(" dB", "")) > 0 else RED
            tk.Label(row, text=value, font=("Consolas", 11, "bold"),
                     bg=BG2, fg=fg_c, anchor="w").pack(side="left", padx=8)

        # ── Metric Explanations ───────────────────────────────────────────
        section_label(scroll_frame, "Score Explanations")
        for name, explanation in self.METRIC_EXPLANATIONS.items():
            exp_card = tk.Frame(scroll_frame, bg=BG2)
            exp_card.pack(fill="x", pady=3)
            exp_inner = tk.Frame(exp_card, bg=BG2)
            exp_inner.pack(fill="x", padx=14, pady=6)
            tk.Label(exp_inner, text=name, font=FONT_B, bg=BG2,
                     fg=ACCENT, anchor="w").pack(anchor="w")
            tk.Label(exp_inner, text=explanation, font=FONT, bg=BG2,
                     fg=FG_DIM, anchor="w", justify="left",
                     wraplength=800).pack(anchor="w", padx=12, pady=(2, 0))

        # ── Close button ──────────────────────────────────────────────────
        footer = tk.Frame(self, bg=BG)
        footer.pack(fill="x", padx=20, pady=10)
        styled_button(footer, "Close", self.destroy, padx=16, pady=8).pack(side="right")
        styled_button(footer, "Copy to Clipboard", lambda: self._copy_results(
            bench_results, aggregate, snr_db, model_type),
            padx=16, pady=8).pack(side="right", padx=8)

    def _copy_results(self, bench_results, aggregate, snr_db, model_type):
        """Copy benchmark results as formatted text to clipboard."""
        lines = [
            f"=== ECG Denoising Benchmark Results ===",
            f"Model: {model_type.upper()}  |  SNR: {snr_db} dB  |  Patients: {len(bench_results)}",
            "",
            "Per-Patient:",
            f"{'Patient':>10} {'Windows':>8} {'SNR In':>8} {'SNR Out':>8} "
            f"{'dSNR':>7} {'PRD In':>8} {'PRD Out':>8} {'MSE In':>10} {'MSE Out':>10}",
            "-" * 90,
        ]
        for r in bench_results:
            m = r["metrics"]
            def _f(v, fmt):
                return "N/A" if _is_nan(v) else format(v, fmt)
            lines.append(
                f"{r['patient']:>10} {r['n_windows']:>8} "
                f"{m['snr_noisy']:>+8.2f} {_f(m['snr_denoised'], '+8.2f'):>8} "
                f"{_f(m['snr_improve'], '+7.2f'):>7} "
                f"{m['prd_noisy']:>8.2f} {_f(m['prd_denoised'], '8.2f'):>8} "
                f"{m['mse_noisy']:>10.6f} {_f(m['mse_denoised'], '10.6f'):>10}"
            )
        lines += [
            "",
            "Aggregate (Average):",
            f"  SNR In:  {aggregate['snr_noisy']:+.2f} dB",
            f"  SNR Out: {aggregate['snr_denoised']:+.2f} dB" if not _is_nan(aggregate['snr_denoised']) else "  SNR Out: N/A",
            f"  dSNR:    {aggregate['snr_improve']:+.2f} dB" if not _is_nan(aggregate['snr_improve']) else "  dSNR:    N/A",
            f"  PRD In:  {aggregate['prd_noisy']:.2f} %",
            f"  PRD Out: {aggregate['prd_denoised']:.2f} %" if not _is_nan(aggregate['prd_denoised']) else "  PRD Out: N/A",
            f"  MSE In:  {aggregate['mse_noisy']:.6f}",
            f"  MSE Out: {aggregate['mse_denoised']:.6f}" if not _is_nan(aggregate['mse_denoised']) else "  MSE Out: N/A",
        ]
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))
        messagebox.showinfo("Copied", "Benchmark results copied to clipboard.", parent=self)


def _is_nan(v):
    return isinstance(v, float) and v != v


# ═════════════════════════════════════════════════════════════════════════════
# Embedded matplotlib ECG Viewer
# ═════════════════════════════════════════════════════════════════════════════

class ECGViewer:
    LEAD_NAMES = ["Lead I", "Lead II"]

    def __init__(self, results, metrics, patient, snr_db, n_windows, container,
                 model=None, device=None, lead_display="both", model_type=None):
        self.results    = results   # dict: clean, noisy, [denoised]
        self.metrics    = metrics
        self.patient    = patient
        self.snr_db     = snr_db
        self.model_type = model_type or "unknown"
        self.n_all      = len(results["clean"])
        self.n_leads    = results["clean"].shape[2]
        self.time_ax    = np.arange(results["clean"].shape[1]) / config.SAMPLING_FREQ
        self.visible    = {k: True for k in results}
        self._model     = model    # may be None (no-model mode)
        self._device    = device

        # Lead display: "both", "1" (Lead I only), "2" (Lead II only)
        self.lead_display   = lead_display
        self.lead_visible   = [True, True]  # [Lead I, Lead II]
        self._apply_lead_display(lead_display)

        step = max(1, self.n_all // n_windows)
        self.overview_idxs = list(range(0, self.n_all, step))[:n_windows]
        self.n_windows     = len(self.overview_idxs)
        self.cur_win       = self.overview_idxs[0]

        # Per-window denoised cache: index → np.ndarray (W, L) or None
        if "denoised" in results:
            # pre-populated from batch inference
            self._win_denoised = {i: results["denoised"][i] for i in range(self.n_all)}
        else:
            self._win_denoised = {}

        self._build(container)
        self._draw_all()

    def _apply_lead_display(self, mode):
        """Set lead_visible flags from display mode string."""
        if mode == "1":
            self.lead_visible = [True, False]
        elif mode == "2":
            self.lead_visible = [False, True]
        else:
            self.lead_visible = [True, True]

    def _build(self, container):
        self.fig = plt.Figure(figsize=(18, 9), facecolor="#070d1a")

        outer = gridspec.GridSpec(
            3, 2,
            figure=self.fig,
            width_ratios=[5, 1.3],
            height_ratios=[4, 1.5, 0.6],
            left=0.04, right=0.98,
            top=0.93, bottom=0.07,
            hspace=0.15, wspace=0.05,
        )

        # Detail view
        detail_gs = gridspec.GridSpecFromSubplotSpec(
            self.n_leads, 1, subplot_spec=outer[0, 0], hspace=0.06,
        )
        self.detail_axes = [self.fig.add_subplot(detail_gs[i]) for i in range(self.n_leads)]

        # Overview strip
        self.overview_ax = self.fig.add_subplot(outer[1, 0])

        # Metrics text bar
        self.bar_ax = self.fig.add_subplot(outer[2, 0])
        self.bar_ax.axis("off")
        self.bar_ax.set_facecolor("#070d1a")

        # Controls panel
        ctrl_gs = gridspec.GridSpecFromSubplotSpec(
            10, 1, subplot_spec=outer[:, 1], hspace=0.5,
        )
        self.ctrl_axes = [self.fig.add_subplot(ctrl_gs[i]) for i in range(10)]
        for ax in self.ctrl_axes:
            ax.set_facecolor("#0a1020")
            for sp in ax.spines.values():
                sp.set_visible(False)

        self._build_controls()

        self.canvas = FigureCanvasTkAgg(self.fig, master=container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _build_controls(self):
        c = self.ctrl_axes

        c[0].text(0.5, 0.5, f"CONTROLS  ({self.model_type.upper()})", ha="center", va="center",
                  fontsize=9, fontweight="bold", color=ACCENT,
                  transform=c[0].transAxes)

        # Signal toggles — fixed: iterate labels, don't use .rectangles
        keys    = list(self.results.keys())
        labels  = [PLOT_LABELS[k] for k in keys]
        actives = [self.visible[k] for k in keys]
        self.chk = CheckButtons(c[1], labels, actives)
        self.chk.ax.set_facecolor("#0a1020")

        # ── FIX: .rectangles removed in mpl ≥ 3.7; style via .labels only ──
        for lbl_obj, key in zip(self.chk.labels, keys):
            lbl_obj.set_color(PLOT_COLORS[key])
            lbl_obj.set_fontsize(7.5)

        # Style the check boxes themselves (works across all mpl versions)
        try:
            # mpl < 3.7
            for rect in self.chk.rectangles:
                rect.set_facecolor("#1e293b")
                rect.set_edgecolor("#334155")
        except AttributeError:
            pass  # mpl ≥ 3.7 — no rectangles attribute; appearance handled by mpl

        self.chk.on_clicked(self._on_toggle)

        # Lead toggles
        lead_labels  = [self.LEAD_NAMES[i] for i in range(self.n_leads)]
        lead_actives = [self.lead_visible[i] for i in range(self.n_leads)]
        self.lead_chk = CheckButtons(c[2], lead_labels, lead_actives)
        self.lead_chk.ax.set_facecolor("#0a1020")
        for lbl_obj in self.lead_chk.labels:
            lbl_obj.set_color("#7ecfff")
            lbl_obj.set_fontsize(7.5)
        try:
            for rect in self.lead_chk.rectangles:
                rect.set_facecolor("#1e293b")
                rect.set_edgecolor("#334155")
        except AttributeError:
            pass
        self.lead_chk.on_clicked(self._on_lead_toggle)

        # Prev / Next buttons
        self.btn_prev = MplButton(c[3], "◄ Prev", color="#1e293b", hovercolor="#334155")
        self.btn_next = MplButton(c[4], "Next ►", color="#1e293b", hovercolor="#334155")
        for b in (self.btn_prev, self.btn_next):
            b.label.set_color("#7ecfff")
            b.label.set_fontsize(9)
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)

        # SNR info
        c[5].text(0.5, 0.7, f"Input noise: {self.snr_db} dB",
                  ha="center", va="top", fontsize=8, color="#94a3b8",
                  transform=c[5].transAxes)

        # Global metrics
        self.metrics_ax = c[6]
        self.metrics_ax.axis("off")
        self.metrics_lbl = self.metrics_ax.text(
            0.05, 0.98, "", va="top", fontsize=7.5, color="#94a3b8",
            transform=self.metrics_ax.transAxes, family="monospace",
        )
        self._refresh_metrics_text()

        # Window info
        self.win_info_ax = c[7]
        self.win_info_ax.axis("off")
        self.win_info_lbl = self.win_info_ax.text(
            0.05, 0.98, "", va="top", fontsize=7.5, color="#64748b",
            transform=self.win_info_ax.transAxes, family="monospace",
        )

        # Run model on current window button
        self.btn_run = MplButton(c[8], "▶ Run Model (this window)",
                                 color="#0f2d1a", hovercolor="#14532d")
        self.btn_run.label.set_color(GREEN)
        self.btn_run.label.set_fontsize(8)
        self.btn_run.on_clicked(self._on_run_window)
        if self._model is None:
            self.btn_run.label.set_color("#334155")
            self.btn_run.ax.set_facecolor("#111827")

        # Keyboard hint
        c[9].text(0.5, 0.5, "← → or buttons to navigate",
                  ha="center", va="center", fontsize=7, color="#334155",
                  transform=c[9].transAxes)

    @torch.no_grad()
    def _on_run_window(self, _event):
        """Run model inference on the current window only and redraw."""
        if self._model is None:
            return
        idx   = self.cur_win
        noisy = self.results["noisy"][idx]   # (W, L)
        x     = torch.from_numpy(noisy[np.newaxis]).to(self._device)  # (1, W, L)
        pred  = self._model(x)
        # Handle embedding model returning (output, commit_loss) tuple
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = pred.cpu().numpy()[0]                                   # (W, L)
        self._win_denoised[idx] = pred

        # Patch the shared denoised array if it exists, otherwise create it lazily
        if "denoised" not in self.results:
            W, L = noisy.shape
            self.results["denoised"] = np.full(
                (self.n_all, W, L), np.nan, dtype=np.float32)
            self.visible["denoised"] = True
        self.results["denoised"][idx] = pred

        # Recompute global metrics from all non-nan windows
        denoised = self.results["denoised"]
        clean    = self.results["clean"]
        mask     = ~np.isnan(denoised[:, 0, 0])   # windows that have been run
        if mask.any():
            self.metrics["snr_denoised"] = _compute_snr(denoised[mask], clean[mask])
            self.metrics["prd_denoised"] = _compute_prd(denoised[mask], clean[mask])
            self.metrics["mse_denoised"] = float(np.mean((denoised[mask] - clean[mask])**2))
            self.metrics["snr_improve"]  = (self.metrics["snr_denoised"]
                                            - self.metrics["snr_noisy"])

        self._refresh_metrics_text()
        self._draw_all()

    def _refresh_metrics_text(self):
        m = self.metrics
        def _nf(v, fmt):
            if isinstance(v, float) and v != v:
                return "   N/A"
            return format(v, fmt)
        lines = [
            "── Global Metrics ──",
            f"SNR in   {_nf(m['snr_noisy'],    '+7.2f')} dB",
            f"SNR out  {_nf(m['snr_denoised'], '+7.2f')} dB",
            f"ΔSNR     {_nf(m['snr_improve'],  '+7.2f')} dB",
            f"PRD in   {_nf(m['prd_noisy'],    '7.2f')} %",
            f"PRD out  {_nf(m['prd_denoised'], '7.2f')} %",
            f"MSE in   {_nf(m['mse_noisy'],    '10.6f')}",
            f"MSE out  {_nf(m['mse_denoised'], '10.6f')}",
        ]
        self.metrics_lbl.set_text("\n".join(lines))

    # ── Draw ──────────────────────────────────────────────────────────────────

    def _draw_detail(self):
        t   = self.time_ax
        idx = self.cur_win
        first_visible_lead = True
        for lead_i, ax in enumerate(self.detail_axes):
            ax.cla()
            ax.set_facecolor("#060c18")
            for sp in ax.spines.values():
                sp.set_edgecolor("#1e293b")

            # Check if this lead is visible
            if lead_i < len(self.lead_visible) and not self.lead_visible[lead_i]:
                ax.set_facecolor("#050a14")
                ax.text(0.5, 0.5, f"{self.LEAD_NAMES[lead_i] if lead_i < 2 else f'Lead {lead_i+1}'}  (hidden)",
                        ha="center", va="center", fontsize=10, color="#1e293b",
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            ax.grid(True, color="#0c1626", linewidth=0.5)

            for key, arr in self.results.items():
                if not self.visible.get(key, True):
                    continue
                lw  = 1.8 if key == "clean" else (0.9 if key == "noisy" else 1.4)
                alp = 1.0 if key == "denoised" else (0.85 if key == "clean" else 0.5)
                ax.plot(t, arr[idx, :, lead_i],
                        color=PLOT_COLORS[key], linewidth=lw, alpha=alp,
                        label=PLOT_LABELS[key] if first_visible_lead else "_")

            ax.set_ylabel(self.LEAD_NAMES[lead_i] if lead_i < 2 else f"Lead {lead_i+1}",
                          color="#475569", fontsize=8)
            ax.yaxis.set_label_position("right")
            ax.tick_params(colors="#334155", labelsize=7)

            if first_visible_lead:
                c_w = self.results["clean"][idx]
                n_w = self.results["noisy"][idx]
                snr_in  = _compute_snr(n_w, c_w)
                info    = f"Window {idx+1}/{self.n_all}  ·  SNR in: {snr_in:.1f} dB"
                if "denoised" in self.results:
                    d_w     = self.results["denoised"][idx]
                    snr_out = _compute_snr(d_w, c_w)
                    prd_in  = _compute_prd(n_w, c_w)
                    prd_out = _compute_prd(d_w, c_w)
                    info = (f"Window {idx+1}/{self.n_all}  ·  "
                            f"SNR: {snr_in:.1f}→{snr_out:.1f} dB "
                            f"(Δ{snr_out-snr_in:+.1f})  ·  "
                            f"PRD: {prd_in:.1f}→{prd_out:.1f}%")
                ax.set_title(info, color="#94a3b8", fontsize=8, pad=3)
                visible_keys = [k for k in self.results if self.visible.get(k, True)]
                handles = [plt.Line2D([0], [0], color=PLOT_COLORS[k], linewidth=1.5)
                           for k in visible_keys]
                lbls = [PLOT_LABELS[k] for k in visible_keys]
                if handles:
                    ax.legend(handles, lbls, loc="upper right", fontsize=7,
                              facecolor="#0a1020", edgecolor="#1e293b", framealpha=0.8)
                first_visible_lead = False

            if lead_i < self.n_leads - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time (s)", color="#475569", fontsize=8)

        # Show which leads are displayed
        visible_lead_names = [self.LEAD_NAMES[i] if i < 2 else f"Lead {i+1}"
                              for i in range(self.n_leads) if self.lead_visible[i]]
        lead_info = ", ".join(visible_lead_names) if visible_lead_names else "None"

        self.win_info_lbl.set_text(
            f"Window  {self.cur_win+1:>5d} / {self.n_all}\n"
            f"Leads: {lead_info}\n"
            f"← → or buttons"
        )
        self.fig.suptitle(
            f"ECG Denoising Test  ·  {self.model_type.upper()}  ·  "
            f"Patient {self.patient}  ·  "
            f"Input noise = {self.snr_db} dB",
            color="#e2e8f0", fontsize=11, y=0.97,
        )

    def _draw_overview(self):
        ax = self.overview_ax
        ax.cla()
        ax.set_facecolor("#04090f")
        ax.grid(True, color="#0c1626", linewidth=0.4)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e293b")

        gap = np.full((10,), np.nan, dtype=np.float32)
        segs = {k: [] for k in self.results}
        tick_pos, tick_labels = [], []
        cursor = 0
        W = self.results["clean"].shape[1]

        for win_idx in self.overview_idxs:
            # Use first visible lead for overview, fallback to 0
            overview_lead = 0
            for li in range(self.n_leads):
                if li < len(self.lead_visible) and self.lead_visible[li]:
                    overview_lead = li
                    break
            for k in self.results:
                segs[k].append(self.results[k][win_idx, :, overview_lead])
                segs[k].append(gap)
            tick_pos.append(cursor + W // 2)
            tick_labels.append(str(win_idx + 1))
            cursor += W + 10

        for key, seg_list in segs.items():
            if not self.visible.get(key, True):
                continue
            lw  = 0.5 if key == "noisy" else 0.9
            alp = 0.4 if key == "noisy" else 0.9
            y = np.concatenate(seg_list)
            x = np.arange(len(y))
            ax.plot(x, y, color=PLOT_COLORS[key], linewidth=lw, alpha=alp)

        # Highlight current window
        if self.cur_win in self.overview_idxs:
            pos = self.overview_idxs.index(self.cur_win)
            xst = pos * (W + 10)
            ax.axvspan(xst, xst + W, color="#7ecfff", alpha=0.08)

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=6, color="#475569")
        ax.tick_params(axis="y", colors="#334155", labelsize=6)
        ax.set_ylabel("Lead I  overview", color="#475569", fontsize=7)
        # Update overview label to show which lead
        overview_lead = 0
        for li in range(self.n_leads):
            if li < len(self.lead_visible) and self.lead_visible[li]:
                overview_lead = li
                break
        lead_name = self.LEAD_NAMES[overview_lead] if overview_lead < 2 else f"Lead {overview_lead+1}"
        ax.set_ylabel(f"{lead_name}  overview", color="#475569", fontsize=7)
        ax.set_xlabel("Window #", color="#475569", fontsize=7)

    def _draw_metrics_bar(self):
        ax = self.bar_ax
        ax.cla()
        ax.axis("off")
        m = self.metrics
        def _nf(v, fmt):
            if isinstance(v, float) and v != v:
                return "N/A"
            return format(v, fmt)
        txt = (
            f"  Global ▸  "
            f"SNR: {_nf(m['snr_noisy'],'.1f')} dB → {_nf(m['snr_denoised'],'.1f')} dB  "
            f"(Δ {_nf(m['snr_improve'],'+.1f')} dB)     "
            f"PRD: {_nf(m['prd_noisy'],'.1f')}% → {_nf(m['prd_denoised'],'.1f')}%     "
            f"MSE: {_nf(m['mse_noisy'],'.5f')} → {_nf(m['mse_denoised'],'.5f')}"
        )
        ax.text(0.01, 0.5, txt, va="center", fontsize=8.5,
                color="#7ecfff", transform=ax.transAxes, family="monospace")

    def _draw_all(self):
        self._draw_detail()
        self._draw_overview()
        self._draw_metrics_bar()
        self.canvas.draw_idle()

    # ── Events ────────────────────────────────────────────────────────────────

    def _on_toggle(self, label):
        for k, lbl_str in PLOT_LABELS.items():
            if lbl_str == label:
                self.visible[k] = not self.visible[k]
        self._draw_all()

    def _on_lead_toggle(self, label):
        """Toggle visibility of a lead."""
        for i, name in enumerate(self.LEAD_NAMES[:self.n_leads]):
            if name == label:
                self.lead_visible[i] = not self.lead_visible[i]
        self._draw_all()

    def _on_prev(self, _):
        if self.cur_win > 0:
            self.cur_win -= 1
            self._draw_all()

    def _on_next(self, _):
        if self.cur_win < self.n_all - 1:
            self.cur_win += 1
            self._draw_all()

    def _on_key(self, event):
        if event.key == "right":
            self._on_next(None)
        elif event.key == "left":
            self._on_prev(None)
        elif event.key == "home":
            self.cur_win = 0; self._draw_all()
        elif event.key == "end":
            self.cur_win = self.n_all - 1; self._draw_all()


# ═════════════════════════════════════════════════════════════════════════════
# Wizard application
# ═════════════════════════════════════════════════════════════════════════════

class ECGTesterApp(tk.Tk):

    STEPS = [StepModel, StepDataSource, StepPatient, StepConfigure,
             StepProcessing, StepResults]

    def __init__(self):
        super().__init__()
        self.title("ECG Denoising Tester")
        self.configure(bg=BG)
        self.minsize(900, 660)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── shared state variables ─────────────────────────────────────────
        self.vars = {
            "checkpoint":    tk.StringVar(value="best"),
            "ckpt_custom":   tk.StringVar(value=""),
            "no_model":      tk.BooleanVar(value=False),
            "model_session": tk.StringVar(value=""),
            "mode":          tk.StringVar(value="A"),
            "split":         tk.StringVar(value="test"),
            "patient":       tk.StringVar(value=""),
            "csv_dir":       tk.StringVar(value=""),
            "raw_dir":       tk.StringVar(value=""),
            "record":        tk.StringVar(value=""),
            "snr":           tk.DoubleVar(value=float(config.SNR_HIGH)),
            "n_windows":     tk.IntVar(value=8),
            "batch_size":    tk.IntVar(value=64),
            "max_windows":   tk.IntVar(value=2000),
            "lead_display":  tk.StringVar(value="both"),
            "cpu_cores":     tk.IntVar(value=3),
            "data_percent":  tk.IntVar(value=20),
            "bench_n_patients": tk.IntVar(value=3),
        }

        self._current_step = 0
        self._steps: list[StepBase] = []
        self._results      = None
        self._metrics      = None
        self._last_model   = None
        self._last_device  = None
        self._model_type   = None

        self._build()
        self._show_step(0)
        self._center()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        # Top accent bar
        tk.Frame(self, bg=ACCENT, height=4).pack(fill="x")

        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(10, 0))
        self._title_lbl = tk.Label(hdr, text="", font=FONT_H, bg=BG, fg=ACCENT)
        self._title_lbl.pack(side="left")
        self._sub_lbl   = tk.Label(hdr, text="", font=FONT, bg=BG, fg=FG_DIM)
        self._sub_lbl.pack(side="left", padx=16)

        # Step pill indicators
        self._pill_frame = tk.Frame(self, bg=BG)
        self._pill_frame.pack(fill="x", padx=24, pady=(6, 0))
        self._pills = []
        for i, cls in enumerate(self.STEPS):
            short = cls.title.split("—")[0].strip()
            lbl_w = tk.Label(self._pill_frame, text=short,
                             font=("Segoe UI", 7), bg=BG2, fg=FG_DIM,
                             padx=8, pady=2, relief="flat")
            lbl_w.pack(side="left", padx=2)
            self._pills.append(lbl_w)
            if i < len(self.STEPS) - 1:
                tk.Label(self._pill_frame, text="›", font=FONT, bg=BG, fg=FG_DIM
                         ).pack(side="left")

        tk.Frame(self, bg=SEP, height=1).pack(fill="x", pady=(8, 0))

        # Content area
        self._content = tk.Frame(self, bg=BG)
        self._content.pack(fill="both", expand=True, padx=24, pady=8)

        # Instantiate all step frames (hidden until needed)
        for cls in self.STEPS:
            step = cls(self._content, self)
            step.place(relwidth=1, relheight=1)
            step.place_forget()
            self._steps.append(step)

        # Footer nav
        tk.Frame(self, bg=SEP, height=1).pack(fill="x")
        footer = tk.Frame(self, bg=BG)
        footer.pack(fill="x", padx=24, pady=10)

        self._btn_back = styled_button(footer, "◄  Back", self._go_back)
        self._btn_back.pack(side="left")

        self._btn_next = styled_button(footer, "Next  ►", self._go_next, accent=True)
        self._btn_next.pack(side="right")

        self._btn_restart = styled_button(footer, "↺  Restart", self._restart)
        self._btn_restart.pack(side="right", padx=8)
        self._btn_restart.pack_forget()

    # ── Step navigation ───────────────────────────────────────────────────────

    def _show_step(self, idx: int):
        for i, step in enumerate(self._steps):
            if i == idx:
                step.place(relwidth=1, relheight=1)
            else:
                step.place_forget()

        self._current_step = idx
        step = self._steps[idx]
        self._title_lbl.config(text=step.title)
        self._sub_lbl.config(text=step.subtitle)

        # Update pills
        for i, pill in enumerate(self._pills):
            if i < idx:
                pill.config(bg=BG2, fg=GREEN)
            elif i == idx:
                pill.config(bg=ACCENT, fg=BG)
            else:
                pill.config(bg=BG2, fg=FG_DIM)

        # Back / Next visibility
        is_first   = idx == 0
        is_proc    = idx == 4  # processing step
        is_results = idx == 5

        self._btn_back.config(state="disabled" if is_first or is_proc else "normal")
        self._btn_next.config(
            text="▶  Run Test" if idx == 3 else
            ("Next  ►" if not is_results else ""),
            state="normal" if not is_proc else "disabled",
        )
        if is_results:
            self._btn_next.pack_forget()
            self._btn_restart.pack(side="right", padx=8)
        else:
            self._btn_next.pack(side="right")
            self._btn_restart.pack_forget()

        step.on_enter()

    def _go_next(self):
        step = self._steps[self._current_step]
        if not step.validate():
            return
        if self._current_step < len(self._steps) - 1:
            self._show_step(self._current_step + 1)

    def _go_back(self):
        if self._current_step > 0:
            self._show_step(self._current_step - 1)

    def _restart(self):
        self._show_step(0)

    # ── Processing ────────────────────────────────────────────────────────────

    def run_processing(self):
        proc_step: StepProcessing = self._steps[4]
        result_step: StepResults  = self._steps[5]
        proc_step.start_spinner()

        def worker():
            try:
                v      = self.vars
                device = torch.device(config.DEVICE)
                log    = proc_step.log

                # Apply CPU core limit
                cpu_cores = int(v["cpu_cores"].get())
                self._apply_cpu_limit(cpu_cores)
                log(f"[device] {device}  (CPU threads: {cpu_cores})")

                # ── Load model ────────────────────────────────────────────
                if not v["no_model"].get():
                    ckpt_key = v["checkpoint"].get()
                    # Resolve checkpoint dir from the selected model session
                    step_model: StepModel = self._steps[0]
                    ckpt_dir = step_model._get_ckpt_dir()
                    if ckpt_key == "best":
                        ckpt_path = str(ckpt_dir / "best.pt")
                    elif ckpt_key == "last":
                        ckpt_path = str(ckpt_dir / "last.pt")
                    else:
                        ckpt_path = v["ckpt_custom"].get().strip()
                    proc_step.set_status("Loading model …")
                    log("\n[1/3] Loading model …")
                    model, model_type = load_model(ckpt_path, device, log)
                    self._model_type = model_type
                else:
                    model = None
                    model_type = None
                    log("\n[1/3] No-model mode — skipping inference.")
                # Store so the viewer can do per-window inference
                self._last_model  = model
                self._last_device = device

                # ── Load data ─────────────────────────────────────────────
                mode    = v["mode"].get()
                split   = v["split"].get()
                patient = v["patient"].get().strip()
                proc_step.set_status("Loading data …")
                log("\n[2/3] Loading data …")

                if mode == "C":
                    patient       = v["record"].get().strip()
                    clean_windows = get_clean_windows_from_wfdb(
                        v["raw_dir"].get(), patient, split, log)
                elif mode == "B":
                    clean_windows = get_clean_windows_from_csv(
                        v["csv_dir"].get(), patient, split, log)
                else:
                    clean_windows = get_clean_windows_from_prepared(patient, split, log)

                log(f"  Windows: {clean_windows.shape}")

                # ── Apply data percentage limit (contiguous to keep time order) ─
                data_pct = int(v["data_percent"].get())
                if data_pct < 100:
                    total_win = len(clean_windows)
                    n_use = max(1, int(total_win * data_pct / 100))
                    clean_windows = clean_windows[:n_use]
                    log(f"  Data limit: {data_pct}% -> using first {n_use}/{total_win} windows (time-sequential)")

                # ── Inference ─────────────────────────────────────────────
                snr  = float(v["snr"].get())
                proc_step.set_status("Running inference …")
                log("\n[3/3] Running inference …")

                # Cap windows to max_windows before inference/noise generation
                max_win = int(v["max_windows"].get())
                if len(clean_windows) > max_win:
                    log(f"  [info] Capping {len(clean_windows)} → {max_win} windows "
                        f"(set Max windows in Step 4 to change)")
                    clean_windows = clean_windows[:max_win]

                if model is not None:
                    results = run_inference(
                        model, clean_windows, snr, device,
                        batch_size=int(v["batch_size"].get()),
                        log=log,
                    )
                else:
                    N_cap = len(clean_windows)
                    W = clean_windows.shape[1]
                    L = clean_windows.shape[2]
                    noisy = np.empty((N_cap, W, L), dtype=np.float32)
                    CHUNK = 256
                    for s in range(0, N_cap, CHUNK):
                        e = min(s + CHUNK, N_cap)
                        for i in range(s, e):
                            noisy[i] = _add_noise(clean_windows[i], snr)
                    results = {"clean": clean_windows, "noisy": noisy}

                metrics = compute_metrics(results)
                log("\n── Metrics ──────────────────────────────────────")
                log(f"  SNR (noisy)    : {metrics['snr_noisy']:+.2f} dB")
                log(f"  SNR (denoised) : {metrics['snr_denoised']:+.2f} dB")
                log(f"  ΔSNR           : {metrics['snr_improve']:+.2f} dB")
                log(f"  PRD (noisy)    : {metrics['prd_noisy']:.2f} %")
                log(f"  PRD (denoised) : {metrics['prd_denoised']:.2f} %")
                log(f"  MSE (noisy)    : {metrics['mse_noisy']:.6f}")
                log(f"  MSE (denoised) : {metrics['mse_denoised']:.6f}")
                log("\n✓ Done.")

                self._results = results
                self._metrics = metrics

                proc_step.stop_spinner()
                proc_step.set_status("✓  Complete — loading results viewer …")

                # Switch to results page (must be on main thread)
                self.after(300, lambda: self._finish_processing(results, metrics, patient, snr))

            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                proc_step.log(f"\n[ERROR] {exc}\n{tb}")
                proc_step.stop_spinner()
                proc_step.set_status(f"✗  Error: {exc}")
                self._btn_back.after(0, self._btn_back.config, {"state": "normal"})

        threading.Thread(target=worker, daemon=True).start()

    def _finish_processing(self, results, metrics, patient, snr):
        result_step: StepResults = self._steps[5]
        result_step.load_results(
            results, metrics, patient, snr,
            model=self._last_model,
            device=self._last_device,
            lead_display=self.vars["lead_display"].get(),
            model_type=self._model_type,
        )
        self._show_step(5)

    # ── CPU core limiting ─────────────────────────────────────────────────

    @staticmethod
    def _apply_cpu_limit(n_cores: int):
        """Restrict PyTorch and numpy to n_cores threads."""
        torch.set_num_threads(n_cores)
        try:
            torch.set_num_interop_threads(max(1, n_cores))
        except RuntimeError:
            pass  # can only be set once, before any parallel work
        os.environ["OMP_NUM_THREADS"] = str(n_cores)
        os.environ["MKL_NUM_THREADS"] = str(n_cores)
        os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)

    # ── Random Multi-Patient Benchmark ────────────────────────────────────

    def run_benchmark(self, status_label: tk.Label, btn: tk.Button):
        """Run the random multi-patient benchmark in a background thread."""

        def _update_status(msg, fg=FG_DIM):
            status_label.after(0, status_label.config, {"text": msg, "fg": fg})

        def worker():
            try:
                v = self.vars
                n_patients = int(v["bench_n_patients"].get())
                snr_db     = float(v["snr"].get())
                data_pct   = int(v["data_percent"].get())
                cpu_cores  = int(v["cpu_cores"].get())
                batch_size = int(v["batch_size"].get())
                split      = v["split"].get()
                mode       = v["mode"].get()
                no_model   = v["no_model"].get()

                # Apply CPU limit
                self._apply_cpu_limit(cpu_cores)
                _update_status(f"CPU threads set to {cpu_cores}")

                device = torch.device(config.DEVICE)

                # ── Load model ────────────────────────────────────────────
                model = None
                model_type = "none"
                if not no_model:
                    ckpt_key = v["checkpoint"].get()
                    sess_name = v["model_session"].get()
                    sessions = _scan_model_sessions()
                    sess = next((s for s in sessions if s["name"] == sess_name), None)
                    ckpt_dir = sess["path"] if sess else config.CHECKPOINT_DIR

                    if ckpt_key == "best":
                        ckpt_path = str(ckpt_dir / "best.pt")
                    elif ckpt_key == "last":
                        ckpt_path = str(ckpt_dir / "last.pt")
                    else:
                        ckpt_path = v["ckpt_custom"].get().strip()

                    _update_status("Loading model ...")
                    model, model_type = load_model(ckpt_path, device, lambda msg, **kw: None)
                    self._model_type = model_type

                # ── Gather available patients ─────────────────────────────
                _update_status("Scanning available patients ...")
                if mode == "A":
                    all_patients = _auto_patients(split)
                elif mode == "B":
                    all_patients = _auto_csv_patients(v["csv_dir"].get(), split)
                elif mode == "C":
                    # For WFDB, scan raw dir for available records
                    raw_dir = Path(v["raw_dir"].get().strip())
                    if raw_dir.exists():
                        all_patients = sorted({
                            p.stem for p in raw_dir.glob("*.hea")
                            if (raw_dir / (p.stem + ".dat")).exists()
                            and not p.stem.endswith("-0")
                        })
                    else:
                        all_patients = []
                else:
                    all_patients = []

                if not all_patients:
                    _update_status("No patients found! Check data source settings.", fg=RED)
                    btn.after(0, btn.config, {"state": "normal", "text": "Run Benchmark"})
                    return

                # Pick random patients
                n_pick = min(n_patients, len(all_patients))
                chosen = random.sample(all_patients, n_pick)
                _update_status(f"Selected {n_pick} patients: {', '.join(chosen)}")

                # ── Run benchmark per patient ─────────────────────────────
                bench_results = []
                t_start = time.time()

                for pi, patient in enumerate(chosen, 1):
                    _update_status(f"[{pi}/{n_pick}] Loading patient {patient} ...")

                    try:
                        if mode == "C":
                            clean_windows = get_clean_windows_from_wfdb(
                                v["raw_dir"].get(), patient, split,
                                lambda msg, **kw: None)
                        elif mode == "B":
                            clean_windows = get_clean_windows_from_csv(
                                v["csv_dir"].get(), patient, split,
                                lambda msg, **kw: None)
                        else:
                            clean_windows = get_clean_windows_from_prepared(
                                patient, split, lambda msg, **kw: None)
                    except Exception as exc:
                        bench_results.append({
                            "patient": patient,
                            "n_windows": 0,
                            "metrics": {k: float("nan") for k in
                                        ["snr_noisy", "snr_denoised", "snr_improve",
                                         "prd_noisy", "prd_denoised",
                                         "mse_noisy", "mse_denoised"]},
                            "error": str(exc),
                        })
                        continue

                    # Apply data percentage limit (contiguous to keep time order)
                    total_win = len(clean_windows)
                    n_use = max(1, int(total_win * data_pct / 100))
                    if n_use < total_win:
                        clean_windows = clean_windows[:n_use]

                    _update_status(
                        f"[{pi}/{n_pick}] Running inference on {patient} "
                        f"({len(clean_windows)}/{total_win} windows) ...")

                    if model is not None:
                        results = run_inference(
                            model, clean_windows, snr_db, device,
                            batch_size=batch_size,
                            log=lambda msg, **kw: None,
                        )
                    else:
                        N_w = len(clean_windows)
                        W = clean_windows.shape[1]
                        L = clean_windows.shape[2]
                        noisy = np.empty((N_w, W, L), dtype=np.float32)
                        for i in range(N_w):
                            noisy[i] = _add_noise(clean_windows[i], snr_db)
                        results = {"clean": clean_windows, "noisy": noisy}

                    metrics = compute_metrics(results)
                    bench_results.append({
                        "patient": patient,
                        "n_windows": len(clean_windows),
                        "metrics": metrics,
                    })

                elapsed = time.time() - t_start

                # ── Compute aggregate metrics ─────────────────────────────
                valid = [r for r in bench_results if r["n_windows"] > 0]
                if valid:
                    aggregate = {}
                    for key in ["snr_noisy", "snr_denoised", "snr_improve",
                                "prd_noisy", "prd_denoised",
                                "mse_noisy", "mse_denoised"]:
                        vals = [r["metrics"][key] for r in valid
                                if not _is_nan(r["metrics"][key])]
                        aggregate[key] = float(np.mean(vals)) if vals else float("nan")
                else:
                    aggregate = {k: float("nan") for k in
                                 ["snr_noisy", "snr_denoised", "snr_improve",
                                  "prd_noisy", "prd_denoised",
                                  "mse_noisy", "mse_denoised"]}

                _update_status(f"Done! {n_pick} patients in {elapsed:.1f}s", fg=GREEN)

                # Show results window on main thread
                self.after(200, lambda: BenchmarkResultsWindow(
                    self, bench_results, aggregate, snr_db, cpu_cores,
                    data_pct, elapsed, model_type))

            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                _update_status(f"Error: {exc}", fg=RED)
                print(f"[Benchmark Error]\n{tb}")

            finally:
                btn.after(0, btn.config, {"state": "normal", "text": "Run Benchmark"})

        threading.Thread(target=worker, daemon=True).start()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _center(self):
        self.update_idletasks()
        w  = self.winfo_width()
        h  = self.winfo_height()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{max(w,1100)}x{max(h,700)}+{(sw-w)//2}+{(sh-h)//2}")

    def _on_close(self):
        plt.close("all")
        self.destroy()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    app = ECGTesterApp()
    app.mainloop()


if __name__ == "__main__":
    main()