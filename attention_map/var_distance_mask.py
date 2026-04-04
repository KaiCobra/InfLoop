"""
var_distance_mask.py — Adaptive side-wave detection & diagonal masking
for VAR (Visual AutoRegressive) self-attention matrices.

When a 2D feature map of shape (H, W) is raster-scanned into a 1D sequence
of length N = H × W, the self-attention matrix A ∈ R^{N×N} develops periodic
side-lobes parallel to the main diagonal, spaced at multiples of W.  These
lobes arise because pixel (i, W−1) and pixel (i+1, 0) are spatial neighbours
in 2D yet sit W positions apart in the 1D ordering.

This module detects the first side-lobe *purely from the attention profile*
(no knowledge of H or W required) and builds a band mask that retains only
the local diagonal context.

Algorithm:
    1. Diagonal profile   m(k) = reduction({A_{ij} : j − i = k})
    2. Gaussian smoothing  m̃(k) = (G_σ ∗ m)(k)
    3. First peak          p₁ = argmax_{k ≥ start_k} [local max of m̃]
    4. First valley        v₁ = argmin_{k > p₁}      [local min of m̃]
    5. Preserve band       b  = v₁

Key improvement over naive implementations: the diagonal profile is computed
via a *composite-key argsort* trick that maps (diagonal_index, value) into a
single float, enabling a fully vectorised median without any Python for-loop.

Usage:
    from attention_map.var_distance_mask import (
        find_first_side_wave_band,
        build_preserve_mask,
        apply_preserve_mask,
        plot_side_wave_detection,
    )
    result = find_first_side_wave_band(A)      # A: (N, N) torch tensor
    A_masked = apply_preserve_mask(A, result["preserve_band"])
    plot_side_wave_detection(result, save_path="side_wave.png")
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple


# ======================================================================
# 1. Diagonal profile
# ======================================================================

def diagonal_profile(
    A: torch.Tensor,
    reduction: str = "median",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the per-diagonal-offset profile of an (N, N) matrix.

    For every signed offset k ∈ [-(N-1), +(N-1)], aggregate all entries
    A[i, j] satisfying j − i = k.

    Args:
        A: Square matrix of shape (N, N).
        reduction: ``"median"`` (robust, via composite-key sort — no Python
            loops) or ``"mean"`` (faster, via ``scatter_add_``).

    Returns:
        ks:      (2N−1,) long tensor of diagonal offsets.
        profile: (2N−1,) float32 tensor of aggregated values.
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1], \
        f"Expected square matrix, got shape {A.shape}"

    N = A.shape[0]
    device = A.device
    num_diags = 2 * N - 1

    # Signed offset matrix: D[i, j] = j − i  ∈ [-(N-1), +(N-1)]
    idx = torch.arange(N, device=device)
    D_signed = idx.unsqueeze(0) - idx.unsqueeze(1)          # (N, N)
    D_shifted = (D_signed + (N - 1)).long()                  # → [0, 2N-2]

    D_flat = D_shifted.reshape(-1)                           # (N²,)
    M_flat = A.reshape(-1).to(torch.float32)                 # (N²,)

    ks = torch.arange(num_diags, device=device, dtype=torch.long) - (N - 1)

    if reduction == "mean":
        # ── Fully vectorised mean via scatter_add_ ──────────────────
        sums = torch.zeros(num_diags, device=device, dtype=torch.float32)
        cnts = torch.zeros(num_diags, device=device, dtype=torch.float32)
        sums.scatter_add_(0, D_flat, M_flat)
        cnts.scatter_add_(0, D_flat, torch.ones_like(M_flat))
        profile = sums / cnts.clamp(min=1)

    elif reduction == "median":
        # ── Fully vectorised median via composite-key sort ──────────
        # Trick: pack (diagonal_index, value) into one float such that
        #   composite = D_shifted  +  normalised_value ∈ [0, 1)
        # A single argsort then groups by diagonal AND sorts by value
        # within each group, so the median can be read off directly.
        M_lo, M_hi = M_flat.min(), M_flat.max()
        denom = (M_hi - M_lo).clamp(min=1e-8)
        M_norm = (M_flat - M_lo) / denom * (1.0 - 1e-6)     # ∈ [0, 1)
        composite = D_flat.float() + M_norm

        order = composite.argsort()
        M_sorted = M_flat[order]

        # Group sizes are deterministic: diagonal k has N − |k| entries
        group_sizes = (N - ks.abs()).long()                   # (2N-1,)
        ends = group_sizes.cumsum(0)
        starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                            ends[:-1]])
        median_pos = starts + group_sizes // 2
        profile = M_sorted[median_pos]

    else:
        raise ValueError(
            f"Unknown reduction '{reduction}'. Use 'median' or 'mean'."
        )

    return ks, profile


# ======================================================================
# 2. Gaussian smoothing
# ======================================================================

def gaussian_smooth_1d(
    x: torch.Tensor,
    sigma: float = 2.5,
) -> torch.Tensor:
    """1D Gaussian smoothing via ``F.conv1d`` with reflect padding.

    Args:
        x: 1D tensor of length L.
        sigma: Standard deviation (σ) of the Gaussian kernel.
               Set ≤ 0 to return *x* unchanged.

    Returns:
        Smoothed 1D float32 tensor of the same length.
    """
    if sigma <= 0:
        return x.clone().float()

    device = x.device
    ks = max(int(6.0 * sigma + 1) | 1, 3)        # kernel width, odd, ≥ 3
    half = ks // 2
    t = torch.arange(ks, device=device, dtype=torch.float32) - half
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()

    x_3d = x.view(1, 1, -1).float()
    x_pad = F.pad(x_3d, (half, half), mode="reflect")
    y = F.conv1d(x_pad, kernel.view(1, 1, -1))
    return y.view(-1)


# ======================================================================
# 3. Side-wave detection
# ======================================================================

def find_first_side_wave_band(
    A: torch.Tensor,
    sigma: float = 2.5,
    min_offset: int = 2,
    reduction: str = "median",
) -> Dict[str, Any]:
    """Detect the first periodic side-wave in a VAR self-attention matrix.

    Steps:
        1. Compute diagonal profile m(k) for k ∈ [-(N-1), +(N-1)].
        2. Gaussian-smooth → m̃(k).
        3. On the k ≥ 0 half, find the first local maximum p₁ at k ≥ min_offset.
        4. Find the first local minimum v₁ at k > p₁.
        5. Set preserve_band = v₁.

    If no side-lobe is detected (e.g. very small matrix), the fallback
    ``preserve_band = ⌊√N⌋`` is used — a reasonable estimate of the 2D
    feature-map width.

    Args:
        A: Self-attention matrix of shape (N, N).
        sigma: Gaussian smoothing σ for the profile curve.
        min_offset: Minimum offset k to begin the peak search (skip the
            trivially high values near the main diagonal).
        reduction: ``"median"`` or ``"mean"`` for the diagonal profile.

    Returns:
        Dictionary with keys::

            ks             : (2N-1,) offset tensor
            profile_raw    : (2N-1,) raw diagonal profile
            profile_smooth : (2N-1,) smoothed profile
            first_peak     : int | None — k of first side-lobe peak
            first_valley   : int | None — k of first valley after peak
            preserve_band  : int — recommended band width
            N              : int — matrix size
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    N = A.shape[0]
    device = A.device

    # ── Profile ──────────────────────────────────────────────────────
    ks, profile_raw = diagonal_profile(A, reduction=reduction)
    profile_smooth = gaussian_smooth_1d(profile_raw, sigma=sigma)

    # ── Analyse k ≥ 0 portion ────────────────────────────────────────
    center = N - 1                                   # index of k = 0 in ks
    pos = profile_smooth[center:]                    # length N, k = 0 … N-1

    first_peak: Optional[int] = None
    first_valley: Optional[int] = None
    preserve_band: int = max(int(N ** 0.5), 1)       # fallback: √N

    if len(pos) > min_offset + 2:
        # Vectorised peak / valley detection on interior points (no loop)
        left  = pos[:-2]                             # m̃(i−1)
        mid   = pos[1:-1]                            # m̃(i)
        right = pos[2:]                              # m̃(i+1)

        # Local maximum: m̃(i) ≥ m̃(i−1) AND m̃(i) > m̃(i+1)
        is_peak = (mid >= left) & (mid > right)
        peak_flags = torch.zeros(len(pos), dtype=torch.bool, device=device)
        peak_flags[1 : len(pos) - 1] = is_peak
        peak_flags[:min_offset] = False

        peak_indices = torch.where(peak_flags)[0]
        if peak_indices.numel() > 0:
            first_peak = peak_indices[0].item()

            # Local minimum: m̃(i) ≤ m̃(i−1) AND m̃(i) < m̃(i+1)
            is_valley = (mid <= left) & (mid < right)
            valley_flags = torch.zeros(len(pos), dtype=torch.bool, device=device)
            valley_flags[1 : len(pos) - 1] = is_valley
            valley_flags[: first_peak + 1] = False  # only after p₁

            valley_indices = torch.where(valley_flags)[0]
            if valley_indices.numel() > 0:
                first_valley = valley_indices[0].item()
                preserve_band = first_valley

    return {
        "ks": ks.cpu(),
        "profile_raw": profile_raw.cpu(),
        "profile_smooth": profile_smooth.cpu(),
        "first_peak": first_peak,
        "first_valley": first_valley,
        "preserve_band": preserve_band,
        "N": N,
    }


# ======================================================================
# 4. Mask construction
# ======================================================================

def build_preserve_mask(
    matrix_size: int,
    preserve_band: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a boolean band mask: ``M[i, j] = (|i − j| ≤ preserve_band)``.

    Args:
        matrix_size: N — size of the square attention matrix.
        preserve_band: Maximum allowed 1D distance |i − j|.
        device: Target device.

    Returns:
        Boolean tensor of shape (N, N).
    """
    idx = torch.arange(matrix_size, device=device)
    D = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
    return D <= preserve_band


# ======================================================================
# 5. Mask application (for post-softmax attention weights)
# ======================================================================

def apply_preserve_mask(
    A: torch.Tensor,
    preserve_band: int,
    renormalize: bool = True,
) -> torch.Tensor:
    """Zero-out entries outside the diagonal band and (optionally) renormalise.

    Designed for *post-softmax* attention weights.  After masking, rows may
    no longer sum to 1; set ``renormalize=True`` to restore row-stochasticity
    (important when the result feeds into SA × CA diffusion).

    Args:
        A: Attention weight matrix, shape ``(N, N)`` or ``(B, H, N, N)``.
        preserve_band: Band width (from ``find_first_side_wave_band``).
        renormalize: Re-normalise each row to sum to 1 after masking.

    Returns:
        Masked (and optionally renormalised) tensor, same shape as *A*.
    """
    is_batched = A.ndim == 4
    if not is_batched:
        A = A.unsqueeze(0).unsqueeze(0)               # → (1, 1, N, N)

    N = A.shape[-1]
    mask = build_preserve_mask(N, preserve_band, device=A.device)  # (N, N)

    A_masked = A * mask[None, None].to(A.dtype)

    if renormalize:
        row_sums = A_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        A_masked = A_masked / row_sums

    if not is_batched:
        A_masked = A_masked.squeeze(0).squeeze(0)

    return A_masked


# ======================================================================
# 6. Visualisation
# ======================================================================

def plot_side_wave_detection(
    result: Dict[str, Any],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 4.5),
) -> None:
    """Visualise the diagonal profile with detected peak / valley markers.

    Args:
        result: Dictionary returned by :func:`find_first_side_wave_band`.
        save_path: If given, save figure to this path; otherwise ``plt.show()``.
        title: Custom plot title.
        figsize: Figure dimensions ``(width, height)`` in inches.
    """
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks     = result["ks"].numpy()
    raw    = result["profile_raw"].numpy()
    smooth = result["profile_smooth"].numpy()
    peak   = result["first_peak"]
    valley = result["first_valley"]
    band   = result["preserve_band"]
    N      = result["N"]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Profiles
    ax.plot(ks, raw, color="#adb5bd", linewidth=0.8, alpha=0.7,
            label="raw profile  m(k)")
    ax.plot(ks, smooth, color="#228be6", linewidth=1.5,
            label="smoothed  m̃(k)")

    # Main diagonal
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4,
               label="main diagonal (k = 0)")

    # Peak
    if peak is not None:
        ax.axvline(peak, color="#fd7e14", linestyle="--", linewidth=1.2,
                   label=f"first peak  p₁ = {peak}")
        ax.axvline(-peak, color="#fd7e14", linestyle="--", linewidth=1.2,
                   alpha=0.35)
        # Annotate peak value
        center = N - 1
        ax.plot(peak, smooth[center + peak], "o", color="#fd7e14",
                markersize=6, zorder=5)

    # Valley
    if valley is not None:
        ax.axvline(valley, color="#e03131", linestyle="--", linewidth=1.2,
                   label=f"first valley  v₁ = {valley}")
        ax.axvline(-valley, color="#e03131", linestyle="--", linewidth=1.2,
                   alpha=0.35)
        ax.plot(valley, smooth[center + valley], "o", color="#e03131",
                markersize=6, zorder=5)

    # Preserve band shading
    ax.axvspan(-band, band, color="#40c057", alpha=0.08,
               label=f"preserve band  b = {band}")
    ax.axvline(band,  color="#40c057", linewidth=2.0)
    ax.axvline(-band, color="#40c057", linewidth=2.0)

    ax.set_xlabel("diagonal offset   k = j − i", fontsize=11)
    ax.set_ylabel("attention value", fontsize=11)
    ax.set_title(title or f"VAR self-attention side-wave detection   (N = {N})",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(ks[0], ks[-1])
    fig.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
