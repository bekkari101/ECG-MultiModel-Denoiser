"""
utils/device.py — Device setup and seed fixing.
"""

import os, sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import random
import numpy as np
import torch
import config


def setup_device():
    """Configure device, thread limits, and determinism. Returns torch.device."""
    if config.DEVICE == "cpu":
        torch.set_num_threads(config.CPU_THREAD_LIMIT)
        torch.set_num_interop_threads(1)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    _fix_seed(config.SEED)

    device = torch.device(config.DEVICE)
    print(f"[device] Using: {device}  |  seed={config.SEED}")
    if config.DEVICE == "cpu":
        print(f"[device] CPU threads capped at {config.CPU_THREAD_LIMIT}")
    return device


def _fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
