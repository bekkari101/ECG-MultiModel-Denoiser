
# 🫀 ECG Denoising with Deep Learning

<div align="center">
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**A comprehensive deep learning framework for ECG signal denoising using rotating-epoch curriculum training on the MIT-BIH Arrhythmia Database.**

[Overview](#-project-overview) · [Architecture](#%EF%B8%8F-model-architectures) · [Training](#-training-strategy) · [Results](#-results--performance) · [Plots](#-training-plots) · [Quick Start](#-quick-start) · [Configuration](#%EF%B8%8F-configuration) · [Structure](#-project-structure)

</div>
---


---

## 📋 Project Overview

Electrocardiogram (ECG) signals recorded in clinical and ambulatory settings are frequently contaminated by various noise sources — muscle artifacts, baseline wander, electrode motion, and electromagnetic interference. This project implements a modular deep learning framework that trains multiple neural network architectures to reconstruct clean ECG signals from their noisy counterparts.

### Core Idea

The system uses a  **rotating-epoch curriculum training strategy** : rather than training on a fixed noise level, the model cycles through clean, low-noise, medium-noise, and high-noise signal pairs in a deliberate sequence. This progressive exposure teaches the network to generalize across the full spectrum of real-world noise conditions without catastrophic forgetting.

### Dataset

All models are trained on the **MIT-BIH Arrhythmia Database** — the gold standard benchmark for ECG research, containing 48 half-hour recordings from 47 subjects sampled at **360 Hz** with two leads.

> Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng Med Biol Mag. 2001;20(3):45-50.

---

## 🏗️ Model Architectures

The framework supports three trained architectures, all sharing the same training pipeline. Every model accepts `(batch, 1440, 2)` tensors and outputs denoised signals of the same shape.

### 1. 🔵 LSTM — Stacked Long Short-Term Memory

Three LSTM layers (`[512, 256, 256]`) process the full 4-second window sequentially, with inter-layer dropout for regularization.

```
Input (B, 1440, 2) → LSTM-512 → Dropout → LSTM-256 → Dropout → LSTM-256 → Linear(2) → Output (B, 1440, 2)
```

**Strengths:** Captures long-range temporal dependencies across the full ECG cycle. Well-suited for P-QRS-T wave pattern recognition.

---

### 2. 🟡 CNN — 1D Convolutional with Residual Blocks

A fully convolutional network with residual skip connections (`[720, 360, 360]` channels). Uses `F.interpolate` to restore original sequence length after pooling.

```
Input → Transpose → Conv1D Projection → ResBlock-720 → Pool → ResBlock-360 → Pool → ResBlock-360 → Upsample → Conv1D Out → Transpose → Output
```

**Strengths:** Highly parallelizable, fast inference. Residual connections prevent vanishing gradients in deeper stacks.

---

### 3. 🔴 Embedding — Tokenized VQ-Transformer *(Current Default)*

Divides the ECG into non-overlapping patches (`PATCH_SIZE=10` samples), projects each into `d_patch=128` space, applies vector quantization against a learned codebook of `1024` entries, then processes the quantized sequence with a Transformer (`d_model=300`, `5 heads`, `5 layers`).

```
Input → Patch Embedding (patch=10) → VQ Codebook (1024 entries) → Transformer (5L, 5H, d=300) → Patch Decoder → Output
                                           ↓
                                Commitment Loss (β=0.25)
```

**Strengths:** Learns a discrete vocabulary of ECG "tokens," enabling structured latent representations. The VQ commitment loss prevents codebook collapse.

---

## 🔄 Training Strategy

### Signal Windowing

| Parameter         | Value        | Description                   |
| ----------------- | ------------ | ----------------------------- |
| `SAMPLING_FREQ` | 360 Hz       | MIT-BIH standard              |
| `WINDOW_SIZE`   | 1440 samples | 4 seconds per window          |
| `WINDOW_SHIFT`  | 45 samples   | 0.125s shift → 96.9% overlap |
| `N_LEADS`       | 2            | Both ECG channels             |

### Data Split

Each patient's signal is divided **70% / 20% / 10%** for train, validation, and test sets before windowing, preventing data leakage.

### Noise Levels

| Epoch Type | SNR             | Clinical Analog                                           |
| ---------- | --------------- | --------------------------------------------------------- |
| `clean`  | ∞ dB           | Reference — identity mapping                             |
| `low`    | **24 dB** | Mild muscle artifact, minor baseline drift                |
| `mid`    | **12 dB** | Moderate clinical interference                            |
| `high`   | **6 dB**  | Severe noise — emergency setting, poor electrode contact |

### Rotating Epoch Curriculum

```
Epoch:  1       2      3      4      5       6      7      8    ...
Type:  clean → low → mid → high → clean → low → mid → high → ...
```

This **prevents catastrophic forgetting** — the model revisits every noise level every 4 epochs. A single global learning rate (`1e-3`) decayed by `ReduceLROnPlateau` governs all epoch types.

```
TOTAL_EPOCHS = 40   →   10 complete rotations of [clean, low, mid, high]
```

### Loss Function

```
Loss = 1.0 × MSE + 0.3 × MAE
```

For the Embedding model, VQ commitment loss (`β=0.25`) is added on top.

---

## 📊 Results & Performance

> **Benchmark on unseen patients** — results from `tools/benchmark_models.py`.
>
> All three models were trained on patients `[105, 106, 220, 222]` and evaluated on **10 unseen patients**: `[101, 107, 108, 109, 111, 117, 118, 121, 217, 231]`.
>
> Each patient contributes **90 non-overlapping windows** (4 s × 360 Hz = 1440 samples). Gaussian noise is added at the specified SNR. All models use the `last.pt` checkpoint.

---

### 🏆 Checkpoint Summary

| Model | Architecture | Epochs | Phase | Val Loss |
|---|:---:|:---:|:---:|:---:|
| **CNN** | 1D-ResNet | 40 / 40 | high | 0.01544 |
| **Embedding** | VQ-Transformer | 40 / 40 | high | 0.01595 |
| **LSTM** | Stacked LSTM | 17 / 40 | clean | 0.00140 |

> LSTM training was interrupted at epoch 17 — it only completed 4 full rotations. CNN and Embedding completed all 10 rotations.

---

### 🟡 Low Noise — 24 dB SNR

| Metric | CNN | Embedding | LSTM | Best |
|---|:---:|:---:|:---:|:---:|
| **SNR Denoised** (dB) | 19.24 | 17.83 | **23.70** | LSTM |
| **SNR Improvement** (dB) | −4.77 | −6.17 | **−0.30** | LSTM |
| **PRD** (%) | 10.91 | 12.84 | **6.53** | LSTM |
| **MSE** | 0.004007 | 0.005548 | **0.001434** | LSTM |
| **RMSE** | 0.0633 | 0.0745 | **0.0379** | LSTM |
| **MAE** | 0.0420 | 0.0485 | **0.0265** | LSTM |
| **Pearson r** | 0.9948 | 0.9941 | **0.9974** | LSTM |

> At low noise all models **degrade** the signal (negative SNR improvement) — the noise is mild enough that the reconstruction error exceeds the noise itself. LSTM degrades it the least, acting closest to an identity mapping.

---

### 🟠 Medium Noise — 12 dB SNR

| Metric | CNN | Embedding | LSTM | Best |
|---|:---:|:---:|:---:|:---:|
| **SNR Denoised** (dB) | **17.38** | 17.19 | 14.15 | CNN |
| **SNR Improvement** (dB) | **+5.39** | +5.18 | +2.15 | CNN |
| **PRD** (%) | **13.51** | 13.83 | 19.62 | CNN |
| **MSE** | **0.006144** | 0.006429 | 0.012942 | CNN |
| **RMSE** | **0.0784** | 0.0802 | 0.1138 | CNN |
| **MAE** | **0.0532** | 0.0553 | 0.0846 | CNN |
| **Pearson r** | 0.9912 | **0.9916** | 0.9762 | Embedding |

> CNN and Embedding are neck-and-neck (< 5% gap on every metric). LSTM lags significantly — its early training stop limited its ability to learn heavy denoising.

---

### 🔴 High Noise — 6 dB SNR

| Metric | CNN | Embedding | LSTM | Best |
|---|:---:|:---:|:---:|:---:|
| **SNR Denoised** (dB) | 15.20 | **15.27** | 9.85 | Embedding |
| **SNR Improvement** (dB) | +9.20 | **+9.27** | +3.85 | Embedding |
| **PRD** (%) | 17.37 | **17.24** | 32.16 | Embedding |
| **MSE** | 0.010147 | **0.010001** | 0.034793 | Embedding |
| **RMSE** | 0.1007 | **0.1000** | 0.1865 | Embedding |
| **MAE** | 0.0707 | **0.0716** | 0.1390 | CNN |
| **Pearson r** | 0.9823 | **0.9828** | 0.9376 | Embedding |

> At severe noise, VQ-Transformer's discrete codebook gives it the edge — quantizing the signal into learned ECG "tokens" filters out noise that falls outside the codebook vocabulary. LSTM recovers less than half the SNR improvement of the other two.

---

### 📌 Head-to-Head Summary

| Noise Condition | 🥇 Winner | 🥈 Runner-up | Key Insight |
|---|:---:|:---:|---|
| **Low (24 dB)** | LSTM | CNN | LSTM barely touches the signal (−0.30 dB); CNN/Embedding over-correct |
| **Medium (12 dB)** | CNN | Embedding | CNN edges Embedding by ~4%; both ≈ +5.3 dB improvement |
| **High (6 dB)** | Embedding | CNN | VQ codebook filters extreme noise; +9.27 dB improvement |

### Key Takeaways

1. **CNN and Embedding are the practical denoising choices** — both deliver +5 to +9 dB SNR improvement at medium-to-high noise, with Pearson > 0.98 even at 6 dB.
2. **LSTM preserves signal integrity best at low noise** but struggles with heavy denoising (only 17 epochs trained; incomplete curriculum exposure to high noise).
3. **No model improves a low-noise signal** — at 24 dB, reconstruction artifacts exceed the noise floor. In production, a noise-level detector should gate whether denoising is applied.
4. **VQ-Transformer wins under severe noise** — its discrete codebook acts as a natural noise filter, projecting noisy inputs onto the nearest clean ECG token.

---

## 📈 Training Plots

All plots are auto-generated during training and saved to `ecg_denoising/outputs/{model}/plots/`.

---

### 🔵 LSTM (lstm1) — 17 / 40 Epochs

| Training Loss | Validation Loss |
|:---:|:---:|
| ![Train Loss](ecg_denoising/outputs/lstm1/plots/train_loss.png) | ![Valid Loss](ecg_denoising/outputs/lstm1/plots/valid_loss.png) |

| Learning Rate | Rotation Summary |
|:---:|:---:|
| ![LR](ecg_denoising/outputs/lstm1/plots/learning_rate.png) | ![Rotation](ecg_denoising/outputs/lstm1/plots/rotation_summary.png) |

| Metrics Grid | SNR Improvement |
|:---:|:---:|
| ![Metrics](ecg_denoising/outputs/lstm1/plots/metrics_grid.png) | ![SNR](ecg_denoising/outputs/lstm1/plots/snr_improvement.png) |

| PRD | Correlation |
|:---:|:---:|
| ![PRD](ecg_denoising/outputs/lstm1/plots/prd.png) | ![Correlation](ecg_denoising/outputs/lstm1/plots/correlation.png) |

---

### 🟡 CNN (cnn1) — 40 / 40 Epochs

| Training Loss | Validation Loss |
|:---:|:---:|
| ![Train Loss](ecg_denoising/outputs/cnn1/plots/train_loss.png) | ![Valid Loss](ecg_denoising/outputs/cnn1/plots/valid_loss.png) |

| Learning Rate | Rotation Summary |
|:---:|:---:|
| ![LR](ecg_denoising/outputs/cnn1/plots/learning_rate.png) | ![Rotation](ecg_denoising/outputs/cnn1/plots/rotation_summary.png) |

| Metrics Grid | SNR Improvement |
|:---:|:---:|
| ![Metrics](ecg_denoising/outputs/cnn1/plots/metrics_grid.png) | ![SNR](ecg_denoising/outputs/cnn1/plots/snr_improvement.png) |

| PRD | Correlation |
|:---:|:---:|
| ![PRD](ecg_denoising/outputs/cnn1/plots/prd.png) | ![Correlation](ecg_denoising/outputs/cnn1/plots/correlation.png) |

---

### 🔴 Embedding / VQ-Transformer (embedding2) — 40 / 40 Epochs

| Training Loss | Validation Loss |
|:---:|:---:|
| ![Train Loss](ecg_denoising/outputs/embedding2/plots/train_loss.png) | ![Valid Loss](ecg_denoising/outputs/embedding2/plots/valid_loss.png) |

| Learning Rate | Rotation Summary |
|:---:|:---:|
| ![LR](ecg_denoising/outputs/embedding2/plots/learning_rate.png) | ![Rotation](ecg_denoising/outputs/embedding2/plots/rotation_summary.png) |

| Metrics Grid | SNR Improvement |
|:---:|:---:|
| ![Metrics](ecg_denoising/outputs/embedding2/plots/metrics_grid.png) | ![SNR](ecg_denoising/outputs/embedding2/plots/snr_improvement.png) |

| PRD | Correlation |
|:---:|:---:|
| ![PRD](ecg_denoising/outputs/embedding2/plots/prd.png) | ![Correlation](ecg_denoising/outputs/embedding2/plots/correlation.png) |

---

## 🚀 Quick Start

### Prerequisites

* Python 3.8+
* NVIDIA GPU with CUDA support (RTX 3080 recommended; 10 GB+ VRAM)
* MIT-BIH Arrhythmia Database

### Installation

```bash
git clone https://github.com/bekkari101/ECG-MultiModel-Denoiser.git
cd ECG-MultiModel-Denoiser
pip install -r requirements.txt
```

Core dependencies: `torch >= 2.0` · `wfdb` · `numpy` · `pandas` · `scikit-learn` · `tqdm` · `matplotlib`

### Dataset Setup

Download from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) and place at:

```
raw/
└── mit-bih-arrhythmia-database-1.0.0/
    ├── RECORDS
    ├── *.atr
    ├── *.dat
    └── *.hea
```

### Data Preparation

```bash
# Recommended: process 5 patients
python ecg_denoising/data/prepare_data_unified.py --n-patients 5

# All available patients
python ecg_denoising/data/prepare_data_unified.py --n-patients 0
```

### Training

```bash
cd ecg_denoising
python main.py
```

Training automatically resumes from `training_state.json` if interrupted.

### Model Selection

Edit `config.py`:

```python
MODEL_TYPE = "embedding"   # Options: lstm | cnn | embedding
```

### Evaluation

```bash
# Interactive GUI tester (single model, visual inspection)
python ecg_denoising/tools/ecg_tester.py

# Headless benchmark (all models × noise levels, outputs JSON)
python ecg_denoising/tools/benchmark_models.py
```

---

## ⚙️ Configuration

All settings in `ecg_denoising/config.py`.

```python
# ── Rotating training ──────────────────────────────────
TOTAL_EPOCHS    = 40
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 100
GRAD_CLIP       = 1.0
MSE_WEIGHT      = 1.0
MAE_WEIGHT      = 0.3

# ── Scheduler ─────────────────────────────────────────
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR   = 0.7
SCHEDULER_MIN_LR   = 1e-6

# ── LSTM ──────────────────────────────────────────────
LSTM_LAYERS  = [512, 256, 256]
LSTM_DROPOUT = 0.1

# ── CNN ───────────────────────────────────────────────
CNN_LAYERS      = [720, 360, 360]
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE   = 2

# ── Embedding ─────────────────────────────────────────
EMBEDDING_VOCAB_SIZE      = 1024
EMBEDDING_PATCH_SIZE      = 10
EMBEDDING_D_MODEL         = 300
EMBEDDING_NHEAD           = 5
EMBEDDING_NUM_LAYERS      = 5
EMBEDDING_COMMITMENT_BETA = 0.25
```

### Patient Selection

```python
PATIENT_FILTER = ["222", "105", "106", "220"]   # specific subset
PATIENT_FILTER = None                            # all available
```

---

## 📁 Project Structure

```
ECG-MultiModel-Denoiser/
├── README.md
├── requirements.txt
├── raw/                                    # MIT-BIH Arrhythmia Database (PhysioNet)
│   └── mit-bih-arrhythmia-database-1.0.0/
│       ├── *.atr  *.hea  *.dat             # 48 patient records @ 360 Hz
│       └── ...
├── scripts/
│   └── convert_and_split.py                # Data conversion & splitting utility
└── ecg_denoising/
    ├── config.py                           # ⚙️  Central configuration
    ├── main.py                             # 🚀 Training entry point
    ├── data/
    │   ├── __init__.py
    │   ├── dataset.py                      # PyTorch Dataset classes
    │   ├── noise.py                        # Gaussian noise generation
    │   ├── windowing.py                    # Sliding window extraction
    │   ├── prepare_data.py                 # Basic data preparation
    │   └── prepare_data_unified.py         # Raw → numpy pipeline (recommended)
    ├── model/
    │   ├── __init__.py
    │   ├── model_factory.py                # Auto-selects architecture from config
    │   ├── lstm_model.py                   # Stacked LSTM architecture
    │   ├── cnn_model.py                    # 1D CNN with residual blocks
    │   ├── embedding_model.py              # VQ-Transformer (default)
    │   ├── rnn_model.py                    # Vanilla RNN architecture
    │   └── transformer_model.py            # Plain Transformer architecture
    ├── training/
    │   ├── __init__.py
    │   ├── trainer.py                      # Rotating-epoch training loop
    │   ├── losses.py                       # MSE + MAE composite loss
    │   ├── metrics.py                      # PRD, SNR, Pearson, Huber
    │   ├── checkpoint.py                   # Save / resume checkpoints
    │   ├── logger.py                       # Training logger
    │   └── plotter.py                      # Loss & metric plots
    ├── utils/
    │   ├── __init__.py
    │   └── device.py                       # CPU / CUDA device selection
    ├── tools/
    │   ├── ecg_tester.py                   # 🖥️  Interactive GUI tester
    │   ├── benchmark_models.py             # 📊 Headless multi-model benchmark
    │   └── outputs/                        # Benchmark JSON results
    │       └── benchmark.json
    └── outputs/                            # Trained model sessions
        ├── cnn1/
        │   ├── checkpoints/
        │   │   ├── best.pt
        │   │   └── last.pt
        │   ├── plots/                      # 8 training plots (see below)
        │   │   ├── train_loss.png
        │   │   ├── valid_loss.png
        │   │   ├── correlation.png
        │   │   ├── prd.png
        │   │   ├── snr_improvement.png
        │   │   ├── learning_rate.png
        │   │   ├── metrics_grid.png
        │   │   └── rotation_summary.png
        │   ├── logs/
        │   └── training_statecnn.json
        ├── embedding2/
        │   ├── checkpoints/
        │   │   ├── best.pt
        │   │   └── last.pt
        │   ├── plots/                      # Same 8 plots
        │   ├── logs/
        │   └── training_state.json
        └── lstm1/
            ├── checkpoints/
            │   ├── best.pt
            │   └── last.pt
            ├── plots/                      # Same 8 plots
            ├── logs/
            └── lstm.json
```

---

## 🖥️ Hardware

| Hardware          | Config               | Notes                                |
| ----------------- | -------------------- | ------------------------------------ |
| CPU               | Intel i9, 10 threads | `CPU_THREAD_LIMIT = 10`            |
| GPU (recommended) | RTX 3080 — 10 GB    | `BATCH_SIZE = 100`fits comfortably |
| GPU (minimum)     | GTX 1070 — 8 GB     | Reduce to `BATCH_SIZE = 64`        |
| GPU (optimal)     | RTX 4090 — 24 GB    | Can increase model size              |

---

## 🔧 Troubleshooting

**CUDA Out of Memory** → reduce `BATCH_SIZE` in `config.py`

**Training won't resume** → verify `training_state.json` exists in the correct session folder

**Unicode errors on Windows**

```bash
set PYTHONIOENCODING=utf-8
```

**Missing dependencies**

```bash
pip install wfdb pandas numpy scikit-learn torch tqdm matplotlib
```

---

## 📚 References

1. Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng Med Biol Mag. 2001.
2. Goldberger AL, et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation. 2000.
3. van den Oord A, et al. *Neural Discrete Representation Learning (VQ-VAE).* NeurIPS 2017.
4. Vaswani A, et al. *Attention Is All You Need.* NeurIPS 2017.

---

## 📄 License

This project is licensed under the **MIT License** — see the [`LICENSE`](LICENSE) file for details.

---

> **Disclaimer:** This project is for research and educational purposes. For clinical deployment, ensure proper validation and regulatory compliance.

<div align="center">
Made with ❤️ for ECG research · Powered by PyTorch · Data from PhysioNet MIT-BIH
</div>
