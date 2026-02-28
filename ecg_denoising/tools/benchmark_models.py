"""
tools/benchmark_models.py — Headless multi-model benchmark (no GUI)
====================================================================
Selects UNSEEN patients (not in config.PATIENT_FILTER), loads every
available model checkpoint, runs inference at LOW / MEDIUM / HIGH noise
levels, and saves all results to a timestamped JSON file.

USAGE:  python -m tools.benchmark_models
        (all settings are constants below — no CLI args needed)

OUTPUT: ecg_denoising/outputs/benchmark_<timestamp>.json
"""

import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ── project root ──────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import config
from model.model_factory import create_model
from data.windowing import prepare_leads
# NOTE: we do NOT import extract_windows — we use non-overlapping windows for testing

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None          # fallback: plain prints

# Supported model types (same as ecg_tester)
SUPPORTED_MODELS = ["lstm", "cnn", "rnn", "transformer", "embedding"]

# ═════════════════════════════════════════════════════════════════════════════
# ██  BENCHMARK CONFIGURATION  — edit these values, then just run the file  ██
# ═════════════════════════════════════════════════════════════════════════════
N_PATIENTS   = 10         # number of unseen patients to test
DATA_PCT     = 100         # % of each patient's windows to use (5-100)
BATCH_SIZE   = 64         # inference batch size
CKPT_KEY     = "last"     # "best" or "last"
SEED         = 42         # random seed for patient selection
CPU_CORES    = 3          # restrict PyTorch to this many CPU threads
# OUTPUT_JSON: set to a path string to override, or None for auto-timestamped
OUTPUT_JSON  = "outputs/benchmark.json"
# ═════════════════════════════════════════════════════════════════════════════

def apply_cpu_limit(n_cores: int = CPU_CORES):
    """Restrict PyTorch + OS threads to *n_cores*."""
    os.environ["OMP_NUM_THREADS"] = str(n_cores)
    os.environ["MKL_NUM_THREADS"] = str(n_cores)
    torch.set_num_threads(n_cores)
    try:
        torch.set_num_interop_threads(n_cores)
    except RuntimeError:
        pass   # can only be set once before parallel work starts

apply_cpu_limit(CPU_CORES)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "raw" / "mit-bih-arrhythmia-database-1.0.0"

def _read_wfdb_record(raw_dir: Path, record: str):
    import wfdb
    rec = wfdb.rdsamp(str(raw_dir / record))
    if isinstance(rec, tuple):
        signals, meta = rec
    else:
        signals = getattr(rec, "p_signals", None) or getattr(rec, "d_signal", None)
        meta = rec.__dict__ if hasattr(rec, "__dict__") else {}
    sig_names = meta.get("sig_name", ["ch1", "ch2"])
    fs = float(meta.get("fs", 360))
    return np.asarray(signals, dtype=np.float32), sig_names, fs


def _split_array(arr, train=0.7, test=0.2):
    n = len(arr)
    t_end = int(n * train)
    v_end = t_end + int(n * test)
    return {"train": arr[:t_end], "test": arr[t_end:v_end], "val": arr[v_end:]}


def _add_noise(clean: np.ndarray, snr_db: float) -> np.ndarray:
    """Vectorised noise addition — works on single windows AND full (N,W,L) batches."""
    # Compute per-window signal power → shape (N,1,1) for batches, scalar for single
    if clean.ndim == 3:
        sig_p = np.mean(clean ** 2, axis=(1, 2), keepdims=True)
    else:
        sig_p = np.mean(clean ** 2)
    n_p = sig_p / (10 ** (snr_db / 10.0) + 1e-12)
    return clean + np.random.randn(*clean.shape).astype(np.float32) * np.sqrt(n_p)


def _compute_snr(pred, target):
    s = np.mean(target ** 2)
    n = np.mean((pred - target) ** 2) + 1e-12
    return float(10 * np.log10(s / n))


def _compute_prd(pred, target):
    return float(100 * np.sqrt(np.sum((pred - target) ** 2) / (np.sum(target ** 2) + 1e-12)))


def _compute_rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def _compute_mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def _compute_pearson(pred, target):
    p = pred.ravel()
    t = target.ravel()
    pm, tm = p.mean(), t.mean()
    num = np.sum((p - pm) * (t - tm))
    den = np.sqrt(np.sum((p - pm) ** 2) * np.sum((t - tm) ** 2)) + 1e-12
    return float(num / den)


# ─────────────────────────────────────────────────────────────────────────────
# Model scanning
# ─────────────────────────────────────────────────────────────────────────────

def scan_model_sessions() -> list[dict]:
    """Return list of model sessions that have checkpoint files."""
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
        model_type = None
        for mt in SUPPORTED_MODELS:
            if item.name.lower().startswith(mt):
                model_type = mt
                break
        has_best = (ckpt_dir / "best.pt").exists()
        has_last = (ckpt_dir / "last.pt").exists()
        if has_best or has_last:
            sessions.append({
                "name": item.name,
                "path": ckpt_dir,
                "model_type": model_type or "unknown",
                "best_exists": has_best,
                "last_exists": has_last,
            })
    return sessions


def _detect_model_type_from_state_dict(state_dict: dict) -> str | None:
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


def load_model(ckpt_path: str, device):
    """Load model from checkpoint, return (model, model_type, ckpt_info)."""
    path = Path(ckpt_path)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Detect type from path, then state dict
    model_type = None
    for part in path.parts:
        for mt in SUPPORTED_MODELS:
            if part.lower().startswith(mt):
                model_type = mt
                break
        if model_type:
            break
    if model_type is None:
        model_type = _detect_model_type_from_state_dict(ckpt.get("model", {}))
    if model_type is None:
        model_type = config.MODEL_TYPE.lower()

    model = create_model(model_type).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    info = {
        "epoch": ckpt.get("epoch", "?"),
        "phase": str(ckpt.get("phase", "?")),
        "val_loss": float(ckpt.get("loss", float("nan"))),
    }
    return model, model_type, info


# ─────────────────────────────────────────────────────────────────────────────
# Patient selection
# ─────────────────────────────────────────────────────────────────────────────

def get_unseen_patients(n: int, seed: int = 42) -> list[str]:
    """
    Pick *n* patients that are NOT in config.PATIENT_FILTER.
    First tries prepared DATA/, then falls back to raw WFDB records.
    """
    # Patients used for training
    seen = set(config.PATIENT_FILTER) if config.PATIENT_FILTER else set()

    # Gather all available patients from prepared data
    available: list[str] = []
    for split in ("train", "val", "test"):
        folder = config.DATA_PROCESSED_DIR / "phase1" / split
        if folder.exists():
            for f in folder.glob("clean_*.npy"):
                pid = f.stem.replace("clean_", "")
                if pid not in seen:
                    available.append(pid)

    # De-duplicate
    available = sorted(set(available))

    # Fallback: scan raw WFDB records if prepared data is empty
    if not available and RAW_DATA_DIR.exists():
        records_file = RAW_DATA_DIR / "RECORDS"
        if records_file.exists():
            with open(records_file, "r") as f:
                all_recs = [ln.strip() for ln in f if ln.strip()]
            available = sorted(r for r in all_recs if r not in seen)

    if not available:
        raise RuntimeError(
            "No unseen patients found. Run prepare_data_unified.py first, "
            "or ensure raw MIT-BIH data is present."
        )

    rng = random.Random(seed)
    n = min(n, len(available))
    chosen = rng.sample(available, n)
    return sorted(chosen)


def _extract_windows_no_overlap(signal: np.ndarray) -> np.ndarray:
    """
    Non-overlapping windowing for testing (shift = window_size).
    Training uses shift=45 (heavy overlap) but that creates way too many
    windows for benchmarking.  Here shift == WINDOW_SIZE → no overlap.
    """
    T, n_leads = signal.shape
    ws = config.WINDOW_SIZE          # 1440 samples (4 s @ 360 Hz)
    if T < ws:
        return np.empty((0, ws, n_leads), dtype=np.float32)
    starts = range(0, T - ws + 1, ws)   # shift = ws → no overlap
    windows = np.stack([signal[s: s + ws] for s in starts], axis=0)
    return windows.astype(np.float32)


def load_patient_windows(patient: str, split: str = "test") -> np.ndarray:
    """Load clean windows for a patient (non-overlapping for fast testing)."""
    # Try prepared data — re-window without overlap
    npy_path = config.DATA_PROCESSED_DIR / "phase1" / split / f"clean_{patient}.npy"
    if npy_path.exists():
        # Prepared data already has overlapping windows; flatten back to
        # continuous signal then re-window without overlap so we don't
        # test on massively redundant data.
        arr = np.load(npy_path).astype(np.float32)
        # Take every Nth window where N = WINDOW_SIZE // WINDOW_SHIFT to
        # approximate non-overlapping (much faster than full re-window)
        stride = max(1, config.WINDOW_SIZE // config.WINDOW_SHIFT)
        return arr[::stride]

    # Fallback: raw WFDB → non-overlapping windows on the fly
    if RAW_DATA_DIR.exists():
        dat_file = RAW_DATA_DIR / f"{patient}.dat"
        hea_file = RAW_DATA_DIR / f"{patient}.hea"
        if dat_file.exists() and hea_file.exists():
            signals, sig_names, fs = _read_wfdb_record(RAW_DATA_DIR, patient)
            splits = _split_array(signals)
            arr = splits.get(split, splits["test"])
            sig = prepare_leads(arr, sig_names)
            windows = _extract_windows_no_overlap(sig)
            return windows.astype(np.float32)

    raise FileNotFoundError(
        f"Cannot find data for patient '{patient}' in prepared DATA/ or raw WFDB."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, windows_clean, snr_db, device, batch_size=64):
    """Run denoising inference, return dict with clean/noisy/denoised arrays."""
    N, W, L = windows_clean.shape

    # Vectorised noise generation (entire array at once — much faster)
    noisy = _add_noise(windows_clean, snr_db)
    denoised = np.empty_like(windows_clean)

    bs = batch_size
    start = 0
    while start < N:
        end = min(start + bs, N)
        try:
            batch = torch.from_numpy(noisy[start:end]).to(device)
            pred = model(batch)
            if isinstance(pred, tuple):
                pred = pred[0]
            denoised[start:end] = pred.cpu().numpy()
            del batch, pred
            start = end
        except RuntimeError as exc:
            if "memory" in str(exc).lower() and bs > 1:
                bs = max(1, bs // 2)
            else:
                raise

    return {"clean": windows_clean, "noisy": noisy, "denoised": denoised}


def compute_all_metrics(results: dict) -> dict:
    """Compute comprehensive metrics from inference results."""
    clean = results["clean"]
    noisy = results["noisy"]
    denoised = results.get("denoised")

    m = {
        "snr_noisy":     _compute_snr(noisy, clean),
        "snr_denoised":  float("nan"),
        "snr_improve":   float("nan"),
        "prd_noisy":     _compute_prd(noisy, clean),
        "prd_denoised":  float("nan"),
        "mse_noisy":     float(np.mean((noisy - clean) ** 2)),
        "mse_denoised":  float("nan"),
        "rmse_noisy":    _compute_rmse(noisy, clean),
        "rmse_denoised": float("nan"),
        "mae_noisy":     _compute_mae(noisy, clean),
        "mae_denoised":  float("nan"),
        "pearson_noisy":     _compute_pearson(noisy, clean),
        "pearson_denoised":  float("nan"),
    }
    if denoised is not None:
        m["snr_denoised"]     = _compute_snr(denoised, clean)
        m["snr_improve"]      = m["snr_denoised"] - m["snr_noisy"]
        m["prd_denoised"]     = _compute_prd(denoised, clean)
        m["mse_denoised"]     = float(np.mean((denoised - clean) ** 2))
        m["rmse_denoised"]    = _compute_rmse(denoised, clean)
        m["mae_denoised"]     = _compute_mae(denoised, clean)
        m["pearson_denoised"] = _compute_pearson(denoised, clean)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Pretty console output
# ─────────────────────────────────────────────────────────────────────────────

_SEP = "─" * 72

def _log(msg: str = ""):
    """Print that works alongside tqdm without corrupting the progress bar."""
    if tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg)

def _fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return "N/A"
        return f"{v:.6f}"
    return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    n_patients: int = 5,
    data_pct: int = 100,
    batch_size: int = 32,
    ckpt_key: str = "best",
    seed: int = 42,
) -> dict:
    """
    Run the full benchmark and return the results dict.

    Returns nested dict:
        {
          "meta": { ... },
          "models": {
            "cnn1":      { "model_type": "cnn", "ckpt_info": {...}, "noise_levels": { "low": {...}, "medium": {...}, "high": {...} } },
            "embedding2": { ... },
            ...
          }
        }
    """
    device = torch.device("cpu")
    noise_levels = {
        "low":    config.SNR_LOW,
        "medium": config.SNR_MEDIUM,
        "high":   config.SNR_HIGH,
    }

    # ── Discover models ───────────────────────────────────────────────────
    sessions = scan_model_sessions()
    usable = []
    for s in sessions:
        pt_file = ckpt_key + ".pt"
        if (s["path"] / pt_file).exists():
            usable.append(s)
    if not usable:
        raise RuntimeError("No model sessions with checkpoint files found in outputs/.")

    _log(f"\n{'═' * 72}")
    _log("  HEADLESS MULTI-MODEL BENCHMARK")
    _log(f"{'═' * 72}")
    _log(f"  Device        : {device}")
    _log(f"  CPU cores     : {CPU_CORES}")
    _log(f"  Checkpoint    : {ckpt_key}.pt")
    _log(f"  Data %        : {data_pct}%")
    _log(f"  Batch size    : {batch_size}")
    _log(f"  Noise levels  : {', '.join(f'{k} ({v} dB)' for k, v in noise_levels.items())}")
    _log(f"  Models found  : {len(usable)}  →  {', '.join(s['name'] for s in usable)}")

    # ── Select unseen patients ────────────────────────────────────────────
    patients = get_unseen_patients(n_patients, seed=seed)
    _log(f"  Patients ({len(patients)}) : {', '.join(patients)}")
    seen = config.PATIENT_FILTER or []
    _log(f"  Seen (excluded): {', '.join(seen) if seen else 'none'}")
    _log(f"{'═' * 72}\n")

    # ── Pre-load patient data ─────────────────────────────────────────────
    _log("Loading patient data ...")
    patient_data: dict[str, np.ndarray] = {}
    for pid in patients:
        try:
            windows = load_patient_windows(pid, split="test")
            # Apply data percentage limit (time-sequential)
            if data_pct < 100:
                n_use = max(1, int(len(windows) * data_pct / 100))
                windows = windows[:n_use]
            patient_data[pid] = windows
            _log(f"  ✓ Patient {pid:>4s}  →  {windows.shape[0]:>5d} windows  shape={windows.shape}")
        except Exception as e:
            _log(f"  ✗ Patient {pid:>4s}  →  FAILED: {e}")
    _log()

    if not patient_data:
        raise RuntimeError("Could not load data for any patient.")

    # ── Benchmark loop ────────────────────────────────────────────────────
    all_results: dict = {
        "meta": {
            "timestamp":   datetime.now().isoformat(),
            "device":      str(device),
            "cpu_cores":   CPU_CORES,
            "checkpoint":  ckpt_key,
            "data_pct":    data_pct,
            "batch_size":  batch_size,
            "noise_levels": {k: float(v) for k, v in noise_levels.items()},
            "patients":    list(patient_data.keys()),
            "seen_patients": list(seen),
            "n_patients":  len(patient_data),
        },
        "models": {},
    }

    total_steps = len(usable) * len(noise_levels) * len(patient_data)
    step_i = 0

    # tqdm outer bar
    if tqdm is not None:
        pbar = tqdm(total=total_steps, desc="Benchmark", unit="test",
                    dynamic_ncols=True, colour="cyan")
    else:
        pbar = None

    for sess in usable:
        model_name = sess["name"]
        ckpt_path = str(sess["path"] / f"{ckpt_key}.pt")

        _log(f"{_SEP}")
        _log(f"  Model: {model_name}  ({sess['model_type'].upper()})")
        _log(f"  Checkpoint: {ckpt_path}")

        try:
            model, model_type, ckpt_info = load_model(ckpt_path, device)
        except Exception as e:
            _log(f"  ✗ Failed to load: {e}")
            if pbar:
                pbar.update(len(noise_levels) * len(patient_data))
            continue

        _log(f"  Loaded OK  epoch={ckpt_info['epoch']}  val_loss={ckpt_info['val_loss']:.6f}")

        model_result = {
            "model_type": model_type,
            "ckpt_info": ckpt_info,
            "noise_levels": {},
        }

        for noise_name, snr_db in noise_levels.items():
            per_patient: list[dict] = []
            agg_clean, agg_noisy, agg_denoised = [], [], []

            for pid, windows in patient_data.items():
                if pbar:
                    pbar.set_postfix_str(f"{model_name} | {noise_name} | P{pid}")
                try:
                    results = run_inference(model, windows, snr_db, device, batch_size)
                    metrics = compute_all_metrics(results)
                    per_patient.append({
                        "patient": pid,
                        "n_windows": int(len(windows)),
                        "metrics": {k: _fmt(v) for k, v in metrics.items()},
                    })
                    agg_clean.append(results["clean"])
                    agg_noisy.append(results["noisy"])
                    if results.get("denoised") is not None:
                        agg_denoised.append(results["denoised"])
                except Exception as e:
                    per_patient.append({
                        "patient": pid,
                        "n_windows": int(len(windows)),
                        "error": str(e),
                    })

                step_i += 1
                if pbar:
                    pbar.update(1)

            # Aggregate metrics across all patients
            aggregate = {}
            if agg_clean and agg_noisy:
                all_clean = np.concatenate(agg_clean, axis=0)
                all_noisy = np.concatenate(agg_noisy, axis=0)
                all_denoised = np.concatenate(agg_denoised, axis=0) if agg_denoised else None
                agg_results = {"clean": all_clean, "noisy": all_noisy}
                if all_denoised is not None:
                    agg_results["denoised"] = all_denoised
                agg_metrics = compute_all_metrics(agg_results)
                aggregate = {k: _fmt(v) for k, v in agg_metrics.items()}

            model_result["noise_levels"][noise_name] = {
                "snr_db": float(snr_db),
                "per_patient": per_patient,
                "aggregate": aggregate,
            }

            # Console summary for this noise level
            if aggregate:
                _log(f"    {noise_name:>6s} ({snr_db:>2.0f} dB)  "
                     f"SNRi={aggregate.get('snr_improve','N/A'):>10s}  "
                     f"PRD={aggregate.get('prd_denoised','N/A'):>10s}  "
                     f"MSE={aggregate.get('mse_denoised','N/A'):>10s}  "
                     f"RMSE={aggregate.get('rmse_denoised','N/A'):>10s}")

        all_results["models"][model_name] = model_result

        # Free model memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if pbar:
        pbar.close()

    _log(f"\n{'═' * 72}")
    _log("  BENCHMARK COMPLETE")
    _log(f"{'═' * 72}\n")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Save to JSON (handles NaN → null)
# ─────────────────────────────────────────────────────────────────────────────

def save_results_json(results: dict, out_path: Path):
    """Write results dict to JSON, converting NaN to null."""
    class _NanEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, float) and math.isnan(obj):
                return None
            if isinstance(obj, Path):
                return str(obj)
            return super().default(obj)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=_NanEncoder)
    _log(f"Results saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    results = run_benchmark(
        n_patients=N_PATIENTS,
        data_pct=DATA_PCT,
        batch_size=BATCH_SIZE,
        ckpt_key=CKPT_KEY,
        seed=SEED,
    )

    if OUTPUT_JSON and isinstance(OUTPUT_JSON, str):
        out_path = Path(OUTPUT_JSON)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = config.BASE_DIR / "outputs" / f"benchmark_{ts}.json"

    save_results_json(results, out_path)


if __name__ == "__main__":
    main()
