"""
config.py — Central configuration for ECG Denoising Framework.

Training Strategy:
    Rotating epoch types: clean → low → mid → high → clean → low → ...
    Each epoch trains on one noise level for one full pass through that loader.
    Continues until TOTAL_EPOCHS is reached — no phases, no stages, no cycles.
"""

import os
import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_DIR           = PROJECT_ROOT / "outputs"
DATA_PROCESSED_DIR = BASE_DIR / "DATA"

# ─────────────────────────────────────────────────────────────────────────────
# Patient Configuration
# ─────────────────────────────────────────────────────────────────────────────
def get_available_patients():
    try:
        if DATA_PROCESSED_DIR.exists():
            phase1_train = DATA_PROCESSED_DIR / "phase1" / "train"
            if phase1_train.exists():
                patients = [f.stem.split('_', 1)[1] for f in phase1_train.glob("clean_*.npy")]
                if patients:
                    return sorted(set(patients))
        if DATA_DIR.exists():
            train_dir = DATA_DIR / "train"
            if train_dir.exists():
                patients = [f.name for f in train_dir.iterdir() if f.is_dir()]
                if patients:
                    return sorted(set(patients))
    except Exception:
        pass
    return ["01", "105", "106", "107", "108", "109", "111", "112", "113", "114",
            "115", "116", "117", "118", "119", "121", "122", "123", "124", "200",
            "201", "202", "203", "205", "207", "208", "209", "210", "212", "213",
            "214", "215", "217", "219", "220", "221", "222", "223", "228", "230",
            "231", "232", "233", "234"]

ALL_AVAILABLE_PATIENTS = get_available_patients()
# PATIENT_FILTER = None   # None = load all available patients
PATIENT_FILTER = ["222", "105", "106", "220"]

# ─────────────────────────────────────────────────────────────────────────────
# Session / Model
# ─────────────────────────────────────────────────────────────────────────────
# Available model types: lstm, cnn, rnn, transformer, embedding
MODEL_TYPE = "embedding"  # Choose from: lstm, cnn, rnn, transformer, embedding 

def get_next_session_id(model_type=None):
    if model_type is None:
        model_type = MODEL_TYPE
    suffix = model_type.lower()
    outputs_dir = BASE_DIR / "outputs"
    if not outputs_dir.exists():
        return 1
    existing = []
    for item in outputs_dir.iterdir():
        if item.is_dir() and item.name.startswith(suffix):
            try:
                existing.append(int(item.name[len(suffix):]))
            except ValueError:
                continue
    return max(existing, default=0) + 1

SESSION_ID = get_next_session_id()

def get_model_output_dir(model_type=None, session_id=1):
    if model_type is None:
        model_type = MODEL_TYPE
    return BASE_DIR / "outputs" / f"{model_type.lower()}{session_id}"

OUTPUT_DIR          = get_model_output_dir(session_id=SESSION_ID)
CHECKPOINT_DIR      = OUTPUT_DIR / "checkpoints"
PLOT_DIR            = OUTPUT_DIR / "plots"
LOG_DIR             = OUTPUT_DIR / "logs"
TRAINING_STATE_PATH = OUTPUT_DIR / "training_state.json"

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
CPU_THREAD_LIMIT = 10
NUM_WORKERS      = 8
PIN_MEMORY       = DEVICE == "cuda"

# ─────────────────────────────────────────────────────────────────────────────
# Signal / Windowing
# ─────────────────────────────────────────────────────────────────────────────
SAMPLING_FREQ  = 360
WINDOW_SECONDS = 4
WINDOW_SIZE    = SAMPLING_FREQ * WINDOW_SECONDS   # 1440 samples
SHIFT_SECONDS  = 0.125
WINDOW_SHIFT   = max(1, int(SHIFT_SECONDS * SAMPLING_FREQ))  # 45 samples
N_LEADS        = 2

# ─────────────────────────────────────────────────────────────────────────────
# Noise / SNR levels
# ─────────────────────────────────────────────────────────────────────────────
SNR_LOW    = 24
SNR_MEDIUM = 12
SNR_HIGH   =  6

NOISE_LEVELS = {"low": SNR_LOW, "medium": SNR_MEDIUM, "high": SNR_HIGH}

# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
TARGET_PARAMETERS = 1_000_000

LSTM_LAYERS   = [512, 256, 256];  LSTM_DROPOUT  = 0.1;  BIDIRECTIONAL = False
CNN_LAYERS    = [720, 360, 360];  CNN_KERNEL_SIZE = 3;   CNN_POOL_SIZE = 2
RNN_LAYERS    = [512, 360, 360];  RNN_DROPOUT   = 0.1
TRANSFORMER_D_MODEL = 128; TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 4; TRANSFORMER_DROPOUT = 0.1

# ── Embedding Model Parameters ─────────────────────────────────────────────
EMBEDDING_VOCAB_SIZE = 1024
EMBEDDING_PATCH_SIZE = 10
EMBEDDING_D_PATCH = 128
EMBEDDING_D_MODEL = 300
EMBEDDING_NHEAD = 5
EMBEDDING_NUM_LAYERS = 5
EMBEDDING_DROPOUT = 0.1
EMBEDDING_COMMITMENT_BETA = 0.25
EMBEDDING_RESIDUAL_HIDDEN = 128

# ─────────────────────────────────────────────────────────────────────────────
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║            ROTATING EPOCH TRAINING                                       ║
# ║                                                                          ║
# ║  Epoch sequence:  clean → low → mid → high → clean → low → mid → high  ║
# ║  …repeats until TOTAL_EPOCHS is exhausted.                              ║
# ║                                                                          ║
# ║  Example: TOTAL_EPOCHS=40 → 10 full rotations of the 4-type pattern.   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# The 4-type rotation order (change order here if desired)
EPOCH_ROTATION = ["clean", "low", "mid", "high"]

# Total epochs to run
TOTAL_EPOCHS = 40

# Single learning rate used throughout (no per-type changes)
LEARNING_RATE = 1e-3

# Batch size
BATCH_SIZE = 100
SEED       = 69

# Gradient clipping
GRAD_CLIP = 1.0

# Loss weights
MSE_WEIGHT = 1.0
MAE_WEIGHT = 0.3

# ReduceLROnPlateau scheduler (applied after every epoch regardless of type)
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR   = 0.7
SCHEDULER_MIN_LR   = 1e-6

# Early stopping — counts any epoch without val-loss improvement (None = off)
EARLY_STOP_PATIENCE = None

# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat stubs (used by prepare_data.py / legacy loaders)
# ─────────────────────────────────────────────────────────────────────────────
PHASE1_EPOCHS = 5;  PHASE1_LR = LEARNING_RATE
PHASE2_EPOCHS = 5;  PHASE2_LR = LEARNING_RATE
PHASE3_EPOCHS = 5;  PHASE3_LR = LEARNING_RATE
PHASE4_EPOCHS = 5;  PHASE4_LR = LEARNING_RATE
PHASE_TRANSITION_WARMUP_EPOCHS = 0
PHASE_TRANSITION_LR_REDUCTION  = 1.0
COMMON_TRAINING_PARAMS = {
    "batch_size": BATCH_SIZE, "window_size": WINDOW_SIZE,
    "n_leads": N_LEADS, "sampling_freq": SAMPLING_FREQ,
}
LOG_EVERY_N_EPOCHS = 1