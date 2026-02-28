"""
model/embedding_model.py — ECG Tokenized Embedding Denoiser.

The core experiment: treat ECG denoising like an LLM sequence task —
quantize the signal into a discrete vocabulary, embed tokens, run a
transformer, then decode back to continuous signal.

Key problems solved vs the naive implementation:
─────────────────────────────────────────────────────────────────────────────
  Problem 1 — argmax is non-differentiable
    → Use Straight-Through Estimator (STE): forward uses hard argmax,
      backward pretends it was an identity (gradients pass through as-if
      no quantization happened). This is how VQ-VAE trains codebooks.

  Problem 2 — codebook collapse (all tokens map to same entry)
    → Commitment loss: penalise the encoder for being far from its
      nearest codebook entry. Keeps the encoder "committed" to codes.
    → EMA codebook update (optional): update codebook entries with
      exponential moving average of assigned encoder outputs instead of
      gradient, more stable than pure backprop through the codebook.

  Problem 3 — tokenization loses continuous detail
    → Residual connection: output = detokenized_reconstruction + residual_correction
      The residual branch (a small CNN) learns the fine-grained details
      that quantization discards. Final output is their sum.

  Problem 4 — single-lead tokenization misses cross-lead correlation
    → Joint tokenization: both leads are tokenized together
      (each codebook entry is an N_LEADS-dimensional vector).

  Problem 5 — positional encoding doesn't capture ECG rhythm structure
    → Learnable relative positional bias (like T5/ALiBi) instead of
      fixed sinusoidal, so the model can learn QRS/P/T-wave spacing.

Architecture overview:
─────────────────────────────────────────────────────────────────────────────

  Input (B, T, L)
      │
      ▼
  ┌─────────────────────────────┐
  │   ECGPatchEncoder           │  Group T samples into patches (like ViT)
  │   Conv1d stride→patch_size  │  Reduces seq length, captures local shape
  └───────────┬─────────────────┘
              │ (B, T//patch, d_patch)
              ▼
  ┌─────────────────────────────┐
  │   VQCodebook (STE)          │  Quantize patch embeddings to vocab tokens
  │   codebook: (V, d_patch)    │  Straight-Through Estimator for grad flow
  └───────────┬─────────────────┘
              │ tokens (B, T//patch) + quantized (B, T//patch, d_patch)
              ▼
  ┌─────────────────────────────┐
  │   Token Embedding           │  d_patch → d_model projection
  │   + Learnable Relative Pos  │  ALiBi-style bias on attention
  └───────────┬─────────────────┘
              │ (B, T//patch, d_model)
              ▼
  ┌─────────────────────────────┐
  │   Transformer Encoder       │  Self-attention across patch tokens
  │   (num_layers, nhead)       │  Learns global ECG structure
  └───────────┬─────────────────┘
              │ (B, T//patch, d_model)
              ▼
  ┌─────────────────────────────┐
  │   PatchDecoder              │  Upsample back to (B, T, L)
  │   ConvTranspose1d           │  Reconstruct continuous signal
  └───────────┬─────────────────┘
              │ (B, T, L)
              ▼
  ┌─────────────────────────────┐   ┌──────────────────────┐
  │   Reconstruction            │ + │   Residual CNN        │  Fine detail
  └─────────────────────────────┘   │   (skip from input)  │  correction
                                    └──────────────────────┘
              │
              ▼
  Output (B, T, L)

Training losses:
  total = task_loss(pred, clean) + λ_commit * commitment_loss
  
  commitment_loss = ||sg[z_e] - e||² + β * ||z_e - sg[e]||²
    where z_e = encoder output, e = nearest codebook entry, sg = stop-gradient

Parameters: ~1.5M (tunable via d_model, num_layers, vocab_size)
"""

import os, sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. Patch Encoder — groups raw samples into patch tokens
# ─────────────────────────────────────────────────────────────────────────────

class PatchEncoder(nn.Module):
    """
    Convert (B, T, L) ECG → (B, T//patch_size, d_patch) patch embeddings.

    Each patch covers `patch_size` time-steps across all leads simultaneously.
    A Conv1d with stride=patch_size is used (non-overlapping patches, like ViT).
    A small stack of depthwise convolutions captures local waveform shape
    within each patch before pooling.

    patch_size=9 → at 360 Hz each patch ≈ 25 ms (roughly one QRS complex width)
    """

    def __init__(self, n_leads: int, d_patch: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

        # Local feature extraction within each patch
        self.local_conv = nn.Sequential(
            nn.Conv1d(n_leads, d_patch // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_patch // 2, d_patch, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Strided conv to produce one vector per patch
        self.patch_proj = nn.Conv1d(
            d_patch, d_patch,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.norm = nn.LayerNorm(d_patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, L) → permute for Conv1d → (B, L, T)
        x = x.permute(0, 2, 1)
        x = self.local_conv(x)       # (B, d_patch, T)
        x = self.patch_proj(x)       # (B, d_patch, T//patch_size)
        x = x.permute(0, 2, 1)       # (B, T//patch_size, d_patch)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. VQ Codebook with Straight-Through Estimator
# ─────────────────────────────────────────────────────────────────────────────

class VQCodebook(nn.Module):
    """
    Vector Quantization with Straight-Through Estimator.

    Forward pass:
      1. Find nearest codebook entry for each patch embedding (hard argmin).
      2. Replace the embedding with the codebook entry value.
      3. STE: in backward pass, gradients flow through as if no quantization
         happened (copy gradients from quantized output to encoder output).

    Also returns commitment_loss so the trainer can add it to the task loss.

    commitment_loss = ||stop_grad(z_e) - e||²   ← codebook update term
                    + β * ||z_e - stop_grad(e)||² ← encoder commitment term
    """

    def __init__(self, vocab_size: int, d_patch: int, commitment_beta: float = 0.25):
        super().__init__()
        self.vocab_size      = vocab_size
        self.d_patch         = d_patch
        self.commitment_beta = commitment_beta

        # Codebook entries
        self.codebook = nn.Embedding(vocab_size, d_patch)
        nn.init.uniform_(self.codebook.weight, -1.0 / vocab_size, 1.0 / vocab_size)

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: (B, S, d_patch)  encoder outputs (patch embeddings)
        Returns:
            z_q:             (B, S, d_patch)  quantized embeddings (STE)
            tokens:          (B, S)           discrete token indices
            commitment_loss: scalar
        """
        B, S, D = z_e.shape
        z_e_flat = z_e.reshape(-1, D)                          # (B*S, D)

        # Compute L2 distances to all codebook entries
        # ||z - e||² = ||z||² + ||e||² - 2 z·e
        z2 = (z_e_flat ** 2).sum(dim=1, keepdim=True)         # (B*S, 1)
        e2 = (self.codebook.weight ** 2).sum(dim=1)            # (V,)
        ze = z_e_flat @ self.codebook.weight.T                 # (B*S, V)
        distances = z2 + e2 - 2 * ze                           # (B*S, V)

        # Hard assignment
        tokens = distances.argmin(dim=1)                        # (B*S,)
        z_q_flat = self.codebook(tokens)                        # (B*S, D)

        # Commitment loss
        commit_loss = (
            F.mse_loss(z_q_flat.detach(), z_e_flat)            # codebook ← encoder
            + self.commitment_beta
            * F.mse_loss(z_q_flat, z_e_flat.detach())          # encoder ← codebook
        )

        # Straight-Through Estimator: copy gradients from z_q to z_e
        z_q_flat_ste = z_e_flat + (z_q_flat - z_e_flat).detach()

        z_q    = z_q_flat_ste.reshape(B, S, D)
        tokens = tokens.reshape(B, S)
        return z_q, tokens, commit_loss

    @torch.no_grad()
    def get_codebook_usage(self, tokens: torch.Tensor) -> dict:
        """Diagnostic: fraction of codebook entries actually used."""
        used  = tokens.unique().numel()
        total = self.vocab_size
        return {"used": used, "total": total, "utilisation": used / total}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Learnable Relative Positional Bias (ALiBi-style)
# ─────────────────────────────────────────────────────────────────────────────

class RelativePositionalBias(nn.Module):
    """
    Learnable scalar bias added to each attention logit based on the
    distance between query and key positions (like ALiBi but learned).

    Replaces fixed sinusoidal PE — the model learns which distances matter
    for ECG (e.g. P→QRS spacing ≈ 200 ms ≈ 8 patches at patch_size=9).
    """

    def __init__(self, nhead: int, max_seq_len: int):
        super().__init__()
        self.nhead = nhead
        # One learnable slope per head
        self.slopes = nn.Parameter(torch.ones(nhead, 1, 1))
        # Relative distance matrix (stays fixed, slopes are learned)
        positions = torch.arange(max_seq_len)
        rel_dist  = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
        self.register_buffer("rel_dist", rel_dist)   # (max_len, max_len)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns bias tensor (nhead, seq_len, seq_len) to add to attn logits."""
        d = self.rel_dist[:seq_len, :seq_len]              # (S, S)
        bias = -self.slopes.abs() * d.unsqueeze(0)         # (H, S, S)
        return bias                                         # broadcast over batch


# ─────────────────────────────────────────────────────────────────────────────
# 4. Transformer with Relative Positional Bias
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLayerWithRPB(nn.Module):
    """
    Standard pre-norm transformer layer with the relative positional
    bias injected into the attention logits.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead,
                                           dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.rpb = RelativePositionalBias(nhead, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # Relative positional bias (nhead, S, S)
        bias = self.rpb(S)                         # (H, S, S)
        bias = bias.unsqueeze(0).expand(B, -1, -1, -1)
        bias = bias.reshape(B * self.attn.num_heads, S, S)

        # Pre-norm attention
        xn = self.norm1(x)
        attn_out, _ = self.attn(xn, xn, xn, attn_mask=bias)
        x = x + attn_out

        # Pre-norm feed-forward
        x = x + self.ff(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. Patch Decoder — reconstructs continuous signal from patch tokens
# ─────────────────────────────────────────────────────────────────────────────

class PatchDecoder(nn.Module):
    """
    (B, S, d_model) → (B, T, L)

    ConvTranspose1d upsamples from S=T//patch_size back to T,
    then refine with local convolutions.
    """

    def __init__(self, d_model: int, n_leads: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

        self.upsample = nn.ConvTranspose1d(
            d_model, d_model // 2,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.refine = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model // 4, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model // 4, n_leads, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_model) → (B, d_model, S)
        x = x.permute(0, 2, 1)
        x = self.upsample(x)   # (B, d_model//2, T)
        x = self.refine(x)     # (B, n_leads, T)
        x = x.permute(0, 2, 1) # (B, T, n_leads)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 6. Residual Fine-Detail Branch
# ─────────────────────────────────────────────────────────────────────────────

class ResidualCNN(nn.Module):
    """
    Small CNN that takes the raw input and produces a residual correction.

    Motivation: quantization discards fine amplitude detail. This branch
    operates directly on the continuous input to learn what the tokenizer
    throws away, then adds it back at the output.

    Keeps parameters small — the transformer handles global structure,
    this handles local micro-corrections.
    """

    def __init__(self, n_leads: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_leads, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, n_leads, kernel_size=1),
        )
        # Zero-init last layer → starts as pure pass-through
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, L)
        h = x.permute(0, 2, 1)   # (B, L, T)
        h = self.net(h)            # (B, L, T)
        return h.permute(0, 2, 1) # (B, T, L)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full Model
# ─────────────────────────────────────────────────────────────────────────────

class ECGTokenizedDenoiser(nn.Module):
    """
    Full ECG Tokenized Embedding Denoiser.

    Args:
        vocab_size       : codebook size (number of discrete ECG tokens)
        patch_size       : samples per patch (9 ≈ 25 ms at 360 Hz)
        d_patch          : patch embedding dimension (= codebook entry dim)
        d_model          : transformer hidden dimension
        nhead            : attention heads
        num_layers       : transformer depth
        dropout          : dropout rate
        commitment_beta  : weight of encoder-to-codebook commitment term
        residual_hidden  : channels in residual fine-detail CNN
        n_leads          : number of ECG leads (from config)
        window_size      : samples per window (from config)
    """

    def __init__(
        self,
        vocab_size:      int   = config.EMBEDDING_VOCAB_SIZE,
        patch_size:      int   = config.EMBEDDING_PATCH_SIZE,
        d_patch:         int   = config.EMBEDDING_D_PATCH,
        d_model:         int   = config.EMBEDDING_D_MODEL,
        nhead:           int   = config.EMBEDDING_NHEAD,
        num_layers:      int   = config.EMBEDDING_NUM_LAYERS,
        dropout:         float = config.EMBEDDING_DROPOUT,
        commitment_beta: float = config.EMBEDDING_COMMITMENT_BETA,
        residual_hidden: int   = config.EMBEDDING_RESIDUAL_HIDDEN,
        n_leads:         int   = config.N_LEADS,
        window_size:     int   = config.WINDOW_SIZE,
    ):
        super().__init__()

        assert window_size % patch_size == 0, (
            f"window_size ({window_size}) must be divisible by patch_size ({patch_size})"
        )

        self.vocab_size      = vocab_size
        self.patch_size      = patch_size
        self.d_patch         = d_patch
        self.d_model         = d_model
        self.n_leads         = n_leads
        self.window_size     = window_size
        self.n_patches       = window_size // patch_size

        # ── Encoder pipeline ─────────────────────────────────────────────
        self.patch_encoder = PatchEncoder(n_leads, d_patch, patch_size)
        self.vq_codebook   = VQCodebook(vocab_size, d_patch, commitment_beta)

        # ── Token → transformer dimension projection ──────────────────────
        self.token_proj = nn.Sequential(
            nn.Linear(d_patch, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Transformer ───────────────────────────────────────────────────
        self.transformer = nn.ModuleList([
            TransformerLayerWithRPB(d_model, nhead, dropout, self.n_patches)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # ── Decoder pipeline ──────────────────────────────────────────────
        self.patch_decoder = PatchDecoder(d_model, n_leads, patch_size)

        # ── Residual fine-detail branch ───────────────────────────────────
        self.residual_cnn = ResidualCNN(n_leads, residual_hidden)

        # ── Output blend weight (learned scalar in [0,1]) ─────────────────
        # alpha=1 → pure transformer output, alpha=0 → pure residual CNN
        self.output_alpha = nn.Parameter(torch.tensor(0.5))

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, L) — noisy ECG input

        Returns:
            output:          (B, T, L) — denoised ECG
            commitment_loss: scalar    — add to task loss during training
        """
        B, T, L = x.shape

        # ── Patch encoding ────────────────────────────────────────────────
        z_e = self.patch_encoder(x)               # (B, S, d_patch)

        # ── Vector quantization (STE) ─────────────────────────────────────
        z_q, tokens, commit_loss = self.vq_codebook(z_e)   # (B, S, d_patch)

        # ── Project to transformer dimension ─────────────────────────────
        h = self.token_proj(z_q)                  # (B, S, d_model)

        # ── Transformer ───────────────────────────────────────────────────
        for layer in self.transformer:
            h = layer(h)
        h = self.final_norm(h)                    # (B, S, d_model)

        # ── Decode back to signal ─────────────────────────────────────────
        recon = self.patch_decoder(h)             # (B, T, L)

        # ── Residual fine-detail correction ──────────────────────────────
        residual = self.residual_cnn(x)           # (B, T, L)

        # ── Blend transformer output + residual ───────────────────────────
        alpha  = torch.sigmoid(self.output_alpha)
        output = alpha * recon + (1.0 - alpha) * residual

        return output, commit_loss

    def encode_to_tokens(self, x: torch.Tensor):
        """
        Inference utility: encode input to discrete token sequence.
        Useful for visualising what the model 'sees' as the ECG vocabulary.
        """
        with torch.no_grad():
            z_e    = self.patch_encoder(x)
            _, tokens, _ = self.vq_codebook(z_e)
        return tokens   # (B, S)

    def codebook_usage(self, x: torch.Tensor) -> dict:
        """Diagnostic: codebook utilisation on a batch."""
        tokens = self.encode_to_tokens(x)
        return self.vq_codebook.get_codebook_usage(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Loss wrapper — combines task loss + commitment loss
# ─────────────────────────────────────────────────────────────────────────────

class TokenizedDenoisingLoss(nn.Module):
    """
    Combined loss for ECGTokenizedDenoiser:
        total = task_loss(pred, clean) + λ_commit * commitment_loss

    task_loss is MSE + β·MAE (same as CombinedLoss in losses.py).
    commitment_loss is returned by the model's forward().
    """

    def __init__(
        self,
        mse_weight:    float = 1.0,
        mae_weight:    float = 0.3,
        commit_weight: float = 0.02,
    ):
        super().__init__()
        self.mse_w    = mse_weight
        self.mae_w    = mae_weight
        self.commit_w = commit_weight

    def forward(
        self,
        pred:           torch.Tensor,
        target:         torch.Tensor,
        commitment_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        task = (
            self.mse_w * F.mse_loss(pred, target)
            + self.mae_w * F.l1_loss(pred, target)
        )
        total = task + self.commit_w * commitment_loss
        return total, {
            "task_loss":   task.item(),
            "commit_loss": commitment_loss.item(),
            "total_loss":  total.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_model_summary(model: ECGTokenizedDenoiser):
    total     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc_p     = sum(p.numel() for p in model.patch_encoder.parameters())
    vq_p      = sum(p.numel() for p in model.vq_codebook.parameters())
    tf_p      = sum(p.numel() for p in model.transformer.parameters())
    dec_p     = sum(p.numel() for p in model.patch_decoder.parameters())
    res_p     = sum(p.numel() for p in model.residual_cnn.parameters())

    print(f"\n{'═'*65}")
    print(f"  ECG Tokenized Embedding Denoiser — Model Summary")
    print(f"{'═'*65}")
    print(f"  Vocab size        : {model.vocab_size}")
    print(f"  Patch size        : {model.patch_size} samples  "
          f"({model.patch_size / config.SAMPLING_FREQ * 1000:.0f} ms @ {config.SAMPLING_FREQ} Hz)")
    print(f"  N patches / window: {model.n_patches}  "
          f"(window={model.window_size} / patch={model.patch_size})")
    print(f"  d_patch           : {model.d_patch}")
    print(f"  d_model           : {model.d_model}")
    print(f"  Transformer       : {len(model.transformer)} layers")
    print(f"  Positional bias   : learnable relative (ALiBi-style)")
    print(f"  Gradient through VQ: Straight-Through Estimator")
    print(f"{'─'*65}")
    print(f"  Patch encoder     : {enc_p:>10,} params")
    print(f"  VQ codebook       : {vq_p:>10,} params")
    print(f"  Transformer       : {tf_p:>10,} params")
    print(f"  Patch decoder     : {dec_p:>10,} params")
    print(f"  Residual CNN      : {res_p:>10,} params")
    print(f"  ─────────────────────────────────────────────")
    print(f"  TOTAL             : {total:>10,} params")
    print(f"{'═'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing ECGTokenizedDenoiser …\n")

    model = ECGTokenizedDenoiser(
        vocab_size  = 512,
        patch_size  = 9,       # 1440 / 9 = 160 patches per window
        d_patch     = 128,
        d_model     = 256,
        nhead       = 8,
        num_layers  = 4,
        dropout     = 0.1,
    )

    B = 4
    x = torch.randn(B, config.WINDOW_SIZE, config.N_LEADS)
    print(f"Input  : {x.shape}  (batch={B}, T={config.WINDOW_SIZE}, leads={config.N_LEADS})")

    output, commit_loss = model(x)
    print(f"Output : {output.shape}")
    print(f"Commit loss: {commit_loss.item():.5f}")

    tokens = model.encode_to_tokens(x)
    print(f"Tokens : {tokens.shape}  (batch={B}, patches={model.n_patches})")
    usage  = model.codebook_usage(x)
    print(f"Codebook usage: {usage['used']}/{usage['total']}  "
          f"({usage['utilisation']*100:.1f}% active)")

    assert output.shape == x.shape, "Shape mismatch!"

    # Test loss
    criterion = TokenizedDenoisingLoss()
    clean     = torch.randn_like(x)
    loss, breakdown = criterion(output, clean, commit_loss)
    print(f"\nLoss breakdown: {breakdown}")

    print_model_summary(model)
    print("✓  All checks passed.")