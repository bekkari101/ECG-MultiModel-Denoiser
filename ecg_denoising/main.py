"""
main.py — Entry point for ECG Denoising Framework.

Rotating epoch training:
    clean → low → mid → high → clean → low → mid → high → ...
    Runs for TOTAL_EPOCHS epochs total.

Usage:
    python main.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import config
from utils.device        import setup_device
from model.model_factory import create_model, print_model_info
from data.dataset        import get_all_loaders
from training.trainer    import run_training


def main():
    device = setup_device()

    print("[main] Auto-detecting available patients …")
    available = config.get_available_patients()
    print(f"[main] Found {len(available)} patients: {available}")
    # PATIENT_FILTER=None in config → all patients will be loaded

    for d in [config.CHECKPOINT_DIR, config.PLOT_DIR, config.LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    model = create_model().to(device)
    print_model_info()
    print(f"[main] Output dir : {config.OUTPUT_DIR}")
    print(f"[main] Rotation   : {' → '.join(config.EPOCH_ROTATION)}")
    print(f"[main] Total epochs: {config.TOTAL_EPOCHS}")

    print("\n[main] Loading all loaders …")
    all_loaders = get_all_loaders()

    print("\n[main] Starting training …\n")
    run_training(model, all_loaders, device)


if __name__ == "__main__":
    main()