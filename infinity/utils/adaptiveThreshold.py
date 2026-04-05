"""
自適應閾值方法模組（Adaptive Threshold Methods）

提供 8 種閾值策略，用於 attention map 二值化：
    1. 固定 percentile
    2. Dynamic threshold（ternary search + reference mask）
    3. Otsu 最大類間方差法
    4. FFT 低通去噪 + Otsu
    5. Spectral Energy Ratio 自適應閾值
    6. Source Image Edge-Attention 跨頻譜相干性
    7. GMM 雙高斯混合模型
    8. 複合方案（方案 6 → Otsu → R_k fallback）

統一入口：
    compute_threshold(filtered_attn, method, low_attn, ...) -> (threshold, denoised_attn, info_str)
"""

from typing import Optional, Tuple
import numpy as np
import cv2


# ── 方法編號常數 ──
METHOD_FIXED_PERCENTILE = 1
METHOD_DYNAMIC_TERNARY = 2
METHOD_OTSU = 3
METHOD_FFT_OTSU = 4
METHOD_SPECTRAL_ENERGY = 5
METHOD_EDGE_COHERENCE = 6
METHOD_GMM = 7
METHOD_COMPOSITE = 8
METHOD_IPR = 9
METHOD_ENTROPY = 10
METHOD_BLOCK_CONSENSUS = 11
METHOD_KNEEDLE = 12

METHOD_NAMES = {
    1: "Fixed Percentile",
    2: "Dynamic Ternary Search",
    3: "Otsu",
    4: "FFT Low-Pass + Otsu",
    5: "Spectral Energy Ratio",
    6: "Edge-Attention Coherence",
    7: "GMM (2-component)",
    8: "Composite (Edge+Otsu+R_k)",
    9: "IPR (Inverse Participation Ratio)",
    10: "Shannon Entropy",
    11: "Block Consensus Voting",
    12: "Kneedle (Elbow Detection)",
}


# =====================================================================
#  方法 1：固定 Percentile
# =====================================================================
def threshold_fixed_percentile(
    filtered_attn: np.ndarray,
    low_attn: bool = False,
    percentile: float = 75.0,
) -> Tuple[float, np.ndarray, str]:
    """
    固定百分位數閾值。

    Args:
        filtered_attn: (H, W) IQR 過濾後的 attention map
        low_attn: True = preserve mask（取低 attention）
        percentile: 閾值百分位數

    Returns:
        (threshold, filtered_attn, info_str)
    """
    if low_attn:
        thr = float(np.percentile(filtered_attn, 100.0 - percentile))
        info = f"fixed percentile (low): thr={thr:.4f}, pct={100.0 - percentile:.0f}"
    else:
        thr = float(np.percentile(filtered_attn, percentile))
        info = f"fixed percentile: thr={thr:.4f}, pct={percentile:.0f}"
    return thr, filtered_attn, info


# =====================================================================
#  方法 2：Dynamic Ternary Search（需 reference mask）
# =====================================================================
def threshold_dynamic_ternary(
    filtered_attn: np.ndarray,
    low_attn: bool = False,
    ref_mask: Optional[np.ndarray] = None,
    max_iters: int = 20,
    fallback_percentile: float = 80.0,
) -> Tuple[float, np.ndarray, str]:
    """
    使用 reference mask 引導的三分搜尋，找到最大化 IoU 的閾值。

    Args:
        filtered_attn: (H, W) attention map
        low_attn: True = preserve mask
        ref_mask: [H_ref, W_ref] bool, True = 編輯區域
        max_iters: 搜尋迭代次數
        fallback_percentile: ref_mask 不可用時的 fallback

    Returns:
        (threshold, filtered_attn, info_str)
    """
    # ref_mask 不可用 → fallback
    if ref_mask is None or ref_mask.sum() == 0 or ref_mask.all():
        thr, attn_out, info = threshold_fixed_percentile(
            filtered_attn, low_attn, fallback_percentile
        )
        return thr, attn_out, f"dynamic fallback → {info}"

    # Resize attention 到 ref_mask 尺寸
    H_ref, W_ref = ref_mask.shape[:2]
    if filtered_attn.shape != (H_ref, W_ref):
        attn_at_ref = cv2.resize(
            filtered_attn.astype(np.float32), (W_ref, H_ref),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        attn_at_ref = filtered_attn.astype(np.float32)

    flat_attn = attn_at_ref.ravel()
    target = (~ref_mask) if low_attn else ref_mask

    def _iou_at_percentile(pct: float) -> float:
        thr = float(np.percentile(flat_attn, pct))
        candidate = (attn_at_ref < thr) if low_attn else (attn_at_ref >= thr)
        inter = float(np.sum(candidate & target))
        union = float(np.sum(candidate | target))
        return inter / union if union > 0 else 0.0

    lo, hi = 0.0, 100.0
    for _ in range(max_iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if _iou_at_percentile(m1) < _iou_at_percentile(m2):
            lo = m1
        else:
            hi = m2

    best_pct = (lo + hi) / 2.0
    best_iou = _iou_at_percentile(best_pct)
    best_thr = float(np.percentile(flat_attn, best_pct))

    # 在原始解析度套用閾值
    thr_orig = float(np.percentile(filtered_attn.ravel(), best_pct))

    # 安全檢查
    if low_attn:
        coverage = float((filtered_attn < thr_orig).mean())
    else:
        coverage = float((filtered_attn >= thr_orig).mean())

    if coverage < 0.01 or best_iou < 0.01:
        thr, attn_out, info = threshold_fixed_percentile(
            filtered_attn, low_attn, fallback_percentile
        )
        return thr, attn_out, (
            f"dynamic failed (IoU={best_iou:.3f}, cov={coverage * 100:.1f}%) → {info}"
        )

    mode_str = "preserve" if low_attn else "focus"
    info = (
        f"dynamic ternary {mode_str}: pct={best_pct:.1f}, "
        f"thr={thr_orig:.4f}, IoU={best_iou:.3f}, cov={coverage * 100:.1f}%"
    )
    return thr_orig, filtered_attn, info


# =====================================================================
#  方法 3：Otsu 最大類間方差法
# =====================================================================
def threshold_otsu(
    filtered_attn: np.ndarray,
    low_attn: bool = False,
    **_kwargs,
) -> Tuple[float, np.ndarray, str]:
    """
    Otsu 自動閾值，無需超參數。

    將 attention map 正規化到 [0, 255] uint8，用 cv2.threshold(OTSU) 計算，
    再反映射回原始值域。
    """
    attn_min = float(filtered_attn.min())
    attn_max = float(filtered_attn.max())

    if attn_max - attn_min < 1e-10:
        # 全部相同值，無法二值化
        thr = float(attn_min)
        return thr, filtered_attn, "otsu: uniform attn, thr=min"

    # 正規化到 [0, 255]
    normed = ((filtered_attn - attn_min) / (attn_max - attn_min) * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(normed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 反映射
    thr = attn_min + (otsu_val / 255.0) * (attn_max - attn_min)

    if low_attn:
        coverage = float((filtered_attn < thr).mean()) * 100
    else:
        coverage = float((filtered_attn >= thr).mean()) * 100

    info = f"otsu: thr={thr:.4f} (otsu_uint8={otsu_val:.0f}), coverage={coverage:.1f}%"
    return thr, filtered_attn, info


# =====================================================================
#  FFT 工具函式
# =====================================================================
def _fft_lowpass(attn: np.ndarray, sigma_f: float) -> np.ndarray:
    """
    2D FFT 高斯低通濾波。

    Args:
        attn: (H, W) float
        sigma_f: 高斯濾波器的 sigma（頻率域）

    Returns:
        (H, W) float — 平滑後的 attention
    """
    H, W = attn.shape
    F_A = np.fft.fft2(attn.astype(np.float64))
    F_A_shifted = np.fft.fftshift(F_A)

    # 建立高斯低通遮罩
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[-cy:H - cy, -cx:W - cx]
    dist_sq = (yy ** 2 + xx ** 2).astype(np.float64)
    gauss = np.exp(-dist_sq / (2.0 * sigma_f ** 2))

    F_filtered = F_A_shifted * gauss
    result = np.fft.ifft2(np.fft.ifftshift(F_filtered)).real

    return result.astype(np.float32)


def _spectral_energy_ratio(attn: np.ndarray, cutoff_ratio: float = 0.25) -> float:
    """
    計算低頻能量佔比 R_k。

    Args:
        attn: (H, W) float
        cutoff_ratio: 截止頻率佔 Nyquist 的比例（預設 0.25）

    Returns:
        R_k ∈ [0, 1]：低頻能量佔比
    """
    H, W = attn.shape
    F_A = np.fft.fft2(attn.astype(np.float64))
    power = np.abs(np.fft.fftshift(F_A)) ** 2

    total_energy = power.sum()
    if total_energy < 1e-15:
        return 0.5  # 無能量，返回中性值

    cy, cx = H // 2, W // 2
    omega_c_y = max(1, int(cy * cutoff_ratio))
    omega_c_x = max(1, int(cx * cutoff_ratio))

    yy, xx = np.ogrid[-cy:H - cy, -cx:W - cx]
    low_freq_mask = ((yy / max(omega_c_y, 1)) ** 2 + (xx / max(omega_c_x, 1)) ** 2) <= 1.0
    low_energy = power[low_freq_mask].sum()

    return float(low_energy / total_energy)


# =====================================================================
#  方法 4：FFT 低通去噪 + Otsu
# =====================================================================
def threshold_fft_otsu(
    filtered_attn: np.ndarray,
    low_attn: bool = False,
    sigma_f: Optional[float] = None,
    **_kwargs,
) -> Tuple[float, np.ndarray, str]:
    """
    先用 FFT 高斯低通去噪，再套用 Otsu 閾值。

    Args:
        sigma_f: 頻率域高斯 sigma。None = 自動（max(H, W) / 4）
    """
    H, W = filtered_attn.shape
    if H < 3 or W < 3:
        # 解析度太低，FFT 無意義，fallback 到 Otsu
        return threshold_otsu(filtered_attn, low_attn)

    if sigma_f is None:
        sigma_f = max(H, W) / 4.0

    denoised = _fft_lowpass(filtered_attn, sigma_f)
    thr, _, otsu_info = threshold_otsu(denoised, low_attn)

    # 用 denoised attention 的閾值，但也存 denoised 供後續使用
    if low_attn:
        coverage = float((denoised < thr).mean()) * 100
    else:
        coverage = float((denoised >= thr).mean()) * 100

    info = f"fft_otsu: σ_f={sigma_f:.1f}, {otsu_info}, coverage={coverage:.1f}%"
    return thr, denoised, info


# =====================================================================
#  方法 5：Spectral Energy Ratio 自適應閾值
# =====================================================================
def threshold_spectral_energy(
    filtered_attn: np.ndarray,
    low_attn: bool = False,
    percentile_min: float = 60.0,
    percentile_max: float = 90.0,
    cutoff_ratio: float = 0.25,
    **_kwargs,
) -> Tuple[float, np.ndarray, str]:
    """
    利用 attention map 的頻譜能量比 R_k 自動調整閾值百分位。

    R_k 大（低頻為主，清晰 blob）→ 更 selective（低 percentile）
    R_k 小（高頻為主，分散噪聲）→ 更保守（高 percentile）
    """
    H, W = filtered_attn.shape
    if H < 3 or W < 3:
        # 解析度太小，直接用中間值
        pct = (percentile_min + percentile_max) / 2.0
        thr = float(np.percentile(filtered_attn, pct))
        return thr, filtered_attn, f"spectral_energy: low res fallback pct={pct:.0f}"

    R_k = _spectral_energy_ratio(filtered_attn, cutoff_ratio)
    # R_k 大 → 低 percentile（更 selective）；R_k 小 → 高 percentile（更保守）
    pct = percentile_max - R_k * (percentile_max - percentile_min)

    if low_attn:
        thr = float(np.percentile(filtered_attn, 100.0 - pct))
        coverage = float((filtered_attn < thr).mean()) * 100
    else:
        thr = float(np.percentile(filtered_attn, pct))
        coverage = float((filtered_attn >= thr).mean()) * 100

    info = (
        f"spectral_energy: R_k={R_k:.3f}, pct={pct:.1f}, "
        f"thr={thr:.4f}, coverage={coverage:.1f}%"
    )
    return thr, filtered_attn, info


# =====================================================================
#  方法 6：Source Image Edge-Attention 跨頻譜相干性
# =====================================================================
def _compute_edge_map(source_image: np.ndarray, spatial_h: int, spatial_w: int) -> np.ndarray:
    """
    計算 source image 的 edge map 並 resize 到 attention 解析度。

    Args:
        source_image: (H_img, W_img, 3) uint8 或 float
        spatial_h, spatial_w: 目標解析度

    Returns:
        (spatial_h, spatial_w) float32, Sobel 梯度模
    """
    if source_image.dtype != np.uint8:
        img = np.clip(source_image, 0, 255).astype(np.uint8)
    else:
        img = source_image

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    # Resize 到 attention 解析度
    gray_resized = cv2.resize(
        gray.astype(np.float32), (spatial_w, spatial_h),
        interpolation=cv2.INTER_AREA,
    )
    # Sobel 梯度
    grad_x = cv2.Sobel(gray_resized, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_resized, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(grad_x ** 2 + grad_y ** 2)

    return edge.astype(np.float32)


def _edge_coherent_filter(filtered_attn: np.ndarray, edge_map: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    跨頻譜相位相干性濾波：保留與 image edge 同相的 attention 頻率成分。

    γ(ω) = Re(F_A · F_E*) / (|F_A| · |F_E| + ε)
    F̃_A = F_A · max(0, γ)
    ã = IFFT2(F̃_A)
    """
    F_A = np.fft.fft2(filtered_attn.astype(np.float64))
    F_E = np.fft.fft2(edge_map.astype(np.float64))

    # 相位相干性
    numerator = np.real(F_A * np.conj(F_E))
    denominator = np.abs(F_A) * np.abs(F_E) + eps
    gamma = numerator / denominator  # [-1, 1]

    # 只保留正相干成分
    weight = np.maximum(0.0, gamma)
    F_filtered = F_A * weight

    result = np.fft.ifft2(F_filtered).real

    # 確保非負（attention 值應 >= 0）
    result = np.maximum(result, 0.0)

    return result.astype(np.float32)


def threshold_edge_coherence(
    filtered_attn: np.ndarray,
    low_attn: bool = False,
    source_image: Optional[np.ndarray] = None,
    **_kwargs,
) -> Tuple[float, np.ndarray, str]:
    """
    Edge-Attention 跨頻譜相干性：
    1. 計算 source image edge map
    2. 用 phase coherence 濾波去除噪聲
    3. Otsu 閾值

    需要 source_image；無提供時 fallback 到 Otsu。
    """
    H, W = filtered_attn.shape

    if source_image is None or H < 3 or W < 3:
        thr, attn_out, info = threshold_otsu(filtered_attn, low_attn)
        return thr, attn_out, f"edge_coherence fallback (no image or low res) → {info}"

    edge_map = _compute_edge_map(source_image, H, W)

    # 檢查 edge_map 是否有效
    if edge_map.max() < 1e-8:
        thr, attn_out, info = threshold_otsu(filtered_attn, low_attn)
        return thr, attn_out, f"edge_coherence fallback (flat edge) → {info}"

    denoised = _edge_coherent_filter(filtered_attn, edge_map)

    # 檢查 denoised 是否退化
    if denoised.max() - denoised.min() < 1e-10:
        thr, attn_out, info = threshold_otsu(filtered_attn, low_attn)
        return thr, attn_out, f"edge_coherence fallback (degenerate) → {info}"

    thr, _, otsu_info = threshold_otsu(denoised, low_attn)

    if low_attn:
        coverage = float((denoised < thr).mean()) * 100
    else:
        coverage = float((denoised >= thr).mean()) * 100

    info = f"edge_coherence + otsu: {otsu_info}, coverage={coverage:.1f}%"
    return thr, denoised, info


# =====================================================================
#  方法 7：GMM 雙高斯混合模型
# =====================================================================
def _fit_gmm_em(
    data: np.ndarray,
    n_iters: int = 50,
    tol: float = 1e-6,
) -> Tuple[float, float, float, float, float, float]:
    """
    用 EM 演算法 fit 2-component GMM。

    Args:
        data: 1D float array
        n_iters: 最大迭代數
        tol: log-likelihood 收斂閾值

    Returns:
        (pi0, mu0, sigma0, pi1, mu1, sigma1)
        其中 mu0 < mu1（component 0 = 背景，component 1 = 前景）
    """
    N = len(data)
    if N < 4:
        mu = float(data.mean())
        return 0.5, mu, 1e-3, 0.5, mu, 1e-3

    # 初始化：用中位數分成兩半
    median = float(np.median(data))
    low = data[data <= median]
    high = data[data > median]

    if len(low) == 0:
        low = data[:N // 2]
    if len(high) == 0:
        high = data[N // 2:]

    mu0 = float(low.mean())
    mu1 = float(high.mean())
    sigma0 = max(float(low.std()), 1e-6)
    sigma1 = max(float(high.std()), 1e-6)
    pi0 = len(low) / N
    pi1 = 1.0 - pi0

    prev_ll = -np.inf
    for _ in range(n_iters):
        # E-step
        log_p0 = np.log(pi0 + 1e-30) - 0.5 * np.log(2 * np.pi * sigma0 ** 2) - (data - mu0) ** 2 / (2 * sigma0 ** 2)
        log_p1 = np.log(pi1 + 1e-30) - 0.5 * np.log(2 * np.pi * sigma1 ** 2) - (data - mu1) ** 2 / (2 * sigma1 ** 2)

        # log-sum-exp for numerical stability
        max_log = np.maximum(log_p0, log_p1)
        log_sum = max_log + np.log(np.exp(log_p0 - max_log) + np.exp(log_p1 - max_log))

        resp0 = np.exp(log_p0 - log_sum)  # responsibility for component 0
        resp1 = 1.0 - resp0

        # log-likelihood
        ll = float(log_sum.sum())
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

        # M-step
        N0 = resp0.sum()
        N1 = resp1.sum()

        if N0 < 1.0 or N1 < 1.0:
            break  # 退化：一個 component 吃掉所有

        pi0 = float(N0 / N)
        pi1 = float(N1 / N)
        mu0 = float((resp0 * data).sum() / N0)
        mu1 = float((resp1 * data).sum() / N1)
        sigma0 = max(float(np.sqrt((resp0 * (data - mu0) ** 2).sum() / N0)), 1e-6)
        sigma1 = max(float(np.sqrt((resp1 * (data - mu1) ** 2).sum() / N1)), 1e-6)

    # 確保 mu0 < mu1
    if mu0 > mu1:
        pi0, pi1 = pi1, pi0
        mu0, mu1 = mu1, mu0
        sigma0, sigma1 = sigma1, sigma0

    return pi0, mu0, sigma0, pi1, mu1, sigma1


def _gmm_decision_boundary(
    pi0: float, mu0: float, sigma0: float,
    pi1: float, mu1: float, sigma1: float,
) -> float:
    """
    計算雙高斯 GMM 的決策邊界（兩個 component 機率密度相等處）。
    使用數值搜尋在 [mu0, mu1] 之間找交叉點。
    """
    from scipy.stats import norm as _norm

    def _diff(x):
        return (pi0 * _norm.pdf(x, mu0, sigma0)
                - pi1 * _norm.pdf(x, mu1, sigma1))

    # 在 [mu0, mu1] 之間線性搜尋
    xs = np.linspace(mu0, mu1, 200)
    diffs = np.array([_diff(x) for x in xs])

    # 找符號變化
    sign_changes = np.where(np.diff(np.sign(diffs)))[0]
    if len(sign_changes) == 0:
        # 無交叉點 → 用中點
        return (mu0 + mu1) / 2.0

    # 取最接近中間的交叉點
    idx = sign_changes[len(sign_changes) // 2]
    # 線性內插
    x0, x1 = xs[idx], xs[idx + 1]
    d0, d1 = diffs[idx], diffs[idx + 1]
    if abs(d1 - d0) < 1e-15:
        return (x0 + x1) / 2.0
    boundary = x0 - d0 * (x1 - x0) / (d1 - d0)
    return float(boundary)


def threshold_gmm(
    filtered_attn: np.ndarray,
    low_attn: bool = False,
    **_kwargs,
) -> Tuple[float, np.ndarray, str]:
    """
    GMM 雙高斯混合模型閾值。

    假設 attention 值服從 background + foreground 雙模態分佈，
    用 EM fit 後取決策邊界作為閾值。
    """
    data = filtered_attn.ravel().astype(np.float64)
    pi0, mu0, sigma0, pi1, mu1, sigma1 = _fit_gmm_em(data)

    # 檢查退化
    if abs(mu1 - mu0) < 1e-8:
        thr, attn_out, info = threshold_otsu(filtered_attn, low_attn)
        return thr, attn_out, f"gmm degenerate (μ0≈μ1) → {info}"

    try:
        thr = _gmm_decision_boundary(pi0, mu0, sigma0, pi1, mu1, sigma1)
    except Exception:
        thr = (mu0 + mu1) / 2.0

    # 確保閾值在合理範圍內
    thr = float(np.clip(thr, filtered_attn.min(), filtered_attn.max()))

    if low_attn:
        coverage = float((filtered_attn < thr).mean()) * 100
    else:
        coverage = float((filtered_attn >= thr).mean()) * 100

    info = (
        f"gmm: π0={pi0:.2f} μ0={mu0:.4f} σ0={sigma0:.4f}, "
        f"π1={pi1:.2f} μ1={mu1:.4f} σ1={sigma1:.4f}, "
        f"thr={thr:.4f}, coverage={coverage:.1f}%"
    )
    return thr, filtered_attn, info


# =====================================================================
#  方法 8：複合方案（Edge-Coherent → Otsu → R_k fallback）
# =====================================================================
def threshold_composite(
    filtered_attn: np.ndarray,
    low_attn: bool = False,
    source_image: Optional[np.ndarray] = None,
    R_min: float = 0.3,
    fallback_percentile: float = 75.0,
    cutoff_ratio: float = 0.25,
    **_kwargs,
) -> Tuple[float, np.ndarray, str]:
    """
    複合方案：
        1. Edge-coherent filtering（需 source_image）
        2. Otsu 閾值
        3. R_k 信心檢查（低於 R_min → fallback 到固定 percentile）

    無 source_image 時跳過步驟 1，直接 FFT 低通 + Otsu。
    """
    H, W = filtered_attn.shape

    # Step 1: Edge-coherent filtering 或 FFT 低通
    if source_image is not None and H >= 3 and W >= 3:
        edge_map = _compute_edge_map(source_image, H, W)
        if edge_map.max() > 1e-8:
            denoised = _edge_coherent_filter(filtered_attn, edge_map)
            filter_type = "edge_coherent"
        else:
            sigma_f = max(H, W) / 4.0
            denoised = _fft_lowpass(filtered_attn, sigma_f)
            filter_type = f"fft_lp(σ={sigma_f:.1f})"
    elif H >= 3 and W >= 3:
        sigma_f = max(H, W) / 4.0
        denoised = _fft_lowpass(filtered_attn, sigma_f)
        filter_type = f"fft_lp(σ={sigma_f:.1f})"
    else:
        denoised = filtered_attn
        filter_type = "none(low_res)"

    # 退化檢查
    if denoised.max() - denoised.min() < 1e-10:
        denoised = filtered_attn
        filter_type += "+fallback_raw"

    # Step 2: R_k 信心度量
    if H >= 3 and W >= 3:
        R_k = _spectral_energy_ratio(denoised, cutoff_ratio)
    else:
        R_k = 0.5

    # Step 3: 根據 R_k 決定策略
    if R_k < R_min:
        # attention 太分散，Otsu 不可靠 → fallback 到固定 percentile
        if low_attn:
            thr = float(np.percentile(filtered_attn, 100.0 - fallback_percentile))
            coverage = float((filtered_attn < thr).mean()) * 100
        else:
            thr = float(np.percentile(filtered_attn, fallback_percentile))
            coverage = float((filtered_attn >= thr).mean()) * 100
        info = (
            f"composite({filter_type}): R_k={R_k:.3f} < R_min={R_min:.2f}, "
            f"fallback pct={fallback_percentile:.0f}, thr={thr:.4f}, coverage={coverage:.1f}%"
        )
        return thr, denoised, info
    else:
        # R_k 夠大，Otsu 可靠
        thr, _, otsu_info = threshold_otsu(denoised, low_attn)
        if low_attn:
            coverage = float((denoised < thr).mean()) * 100
        else:
            coverage = float((denoised >= thr).mean()) * 100
        info = (
            f"composite({filter_type}): R_k={R_k:.3f} ≥ R_min={R_min:.2f}, "
            f"{otsu_info}, coverage={coverage:.1f}%"
        )
        return thr, denoised, info


# =====================================================================
#  方法 9：IPR（Inverse Participation Ratio）自適應面積估計
# =====================================================================
def threshold_ipr(
    filtered_attn: np.ndarray,
    low_attn: bool = False,
    shrink_gamma: float = 1.0,
    **_kwargs,
) -> Tuple[float, np.ndarray, str]:
    """
    利用逆參與率（IPR）估計 attention 的等效集中面積，再反推 percentile。

    f_hat = (sum A)^2 / (N * sum A^2)
    percentile = clip(1 - gamma * f_hat, 0.50, 0.98) * 100

    小物件 → f_hat 小 → percentile 高（~95）；大物件 → f_hat 大 → percentile 低（~70）。

    Args:
        shrink_gamma: 收縮係數（預設 1.0 = 不收縮；<1 偏向更 tight 的 mask）
    """
    sum_A = float(filtered_attn.sum())
    sum_A2 = float((filtered_attn ** 2).sum())
    N = filtered_attn.size

    if sum_A2 < 1e-15:
        # 全零 attention
        pct = 75.0
        f_hat = 0.0
    else:
        f_hat = (sum_A ** 2) / (N * sum_A2)
        pct = float(np.clip((1.0 - shrink_gamma * f_hat) * 100.0, 50.0, 98.0))

    if low_attn:
        thr = float(np.percentile(filtered_attn, 100.0 - pct))
        coverage = float((filtered_attn < thr).mean()) * 100
    else:
        thr = float(np.percentile(filtered_attn, pct))
        coverage = float((filtered_attn >= thr).mean()) * 100

    info = (
        f"ipr: f̂={f_hat:.4f}, γ={shrink_gamma:.2f}, pct={pct:.1f}, "
        f"thr={thr:.4f}, coverage={coverage:.1f}%"
    )
    return thr, filtered_attn, info


# =====================================================================
#  方法 10：Shannon Entropy 有效面積估計
# =====================================================================
def threshold_entropy(
    filtered_attn: np.ndarray,
    low_attn: bool = False,
    **_kwargs,
) -> Tuple[float, np.ndarray, str]:
    """
    用 Shannon 熵估計 attention 的等效展開面積。

    p_ij = A_ij / sum(A)
    H = -sum(p * ln(p))
    N_eff = exp(H)
    f_hat = N_eff / N
    percentile = clip(1 - f_hat, 0.50, 0.98) * 100
    """
    N = filtered_attn.size
    sum_A = float(filtered_attn.sum())

    if sum_A < 1e-15:
        pct = 75.0
        f_hat = 0.0
        N_eff = 0.0
    else:
        p = filtered_attn.ravel().astype(np.float64) / sum_A
        # 避免 log(0)
        p_safe = p[p > 1e-30]
        H = -float(np.sum(p_safe * np.log(p_safe)))
        N_eff = float(np.exp(H))
        f_hat = N_eff / N
        pct = float(np.clip((1.0 - f_hat) * 100.0, 50.0, 98.0))

    if low_attn:
        thr = float(np.percentile(filtered_attn, 100.0 - pct))
        coverage = float((filtered_attn < thr).mean()) * 100
    else:
        thr = float(np.percentile(filtered_attn, pct))
        coverage = float((filtered_attn >= thr).mean()) * 100

    info = (
        f"entropy: N_eff={N_eff:.1f}, f̂={f_hat:.4f}, pct={pct:.1f}, "
        f"thr={thr:.4f}, coverage={coverage:.1f}%"
    )
    return thr, filtered_attn, info


# =====================================================================
#  方法 11：Block Consensus Voting（逐 block Otsu 投票）
# =====================================================================
def threshold_block_consensus(
    filtered_attn: np.ndarray,
    low_attn: bool = False,
    attn_stack: Optional[np.ndarray] = None,
    **_kwargs,
) -> Tuple[float, np.ndarray, str]:
    """
    對 IQR 過濾後保留的每個 block 獨立做 Otsu 二值化，
    計算投票率 V(i,j)（多少比例 block 認為該位置是 focus），
    以面積中位數估計物件大小，反推 percentile。

    若無 attn_stack（fallback），退化為對 filtered_attn 做 IPR。

    Args:
        attn_stack: (B, H, W) ndarray — IQR 過濾後保留的各 block attention maps
    """
    N = filtered_attn.size

    if attn_stack is None or len(attn_stack) < 2:
        # fallback 到 IPR
        thr, attn_out, info = threshold_ipr(filtered_attn, low_attn)
        return thr, attn_out, f"block_consensus fallback (no stack) → {info}"

    B = attn_stack.shape[0]
    block_masks = []
    block_areas = []

    for b in range(B):
        a_b = attn_stack[b]
        a_min, a_max = float(a_b.min()), float(a_b.max())
        if a_max - a_min < 1e-10:
            # 均勻值 → 全 False
            m_b = np.zeros_like(a_b, dtype=bool)
        else:
            normed = ((a_b - a_min) / (a_max - a_min) * 255).astype(np.uint8)
            otsu_val, _ = cv2.threshold(normed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_thr = a_min + (otsu_val / 255.0) * (a_max - a_min)
            m_b = a_b >= otsu_thr
        block_masks.append(m_b)
        block_areas.append(float(m_b.mean()))

    # 投票率 V(i,j)
    vote_map = np.mean(np.stack(block_masks, axis=0).astype(np.float32), axis=0)  # (H, W)

    # 面積中位數
    f_median = float(np.median(block_areas))
    f_hat = float(np.clip(f_median, 0.01, 0.99))
    pct = float(np.clip((1.0 - f_hat) * 100.0, 50.0, 98.0))

    if low_attn:
        thr = float(np.percentile(filtered_attn, 100.0 - pct))
        coverage = float((filtered_attn < thr).mean()) * 100
    else:
        thr = float(np.percentile(filtered_attn, pct))
        coverage = float((filtered_attn >= thr).mean()) * 100

    areas_str = ", ".join(f"{a:.2f}" for a in block_areas[:5])
    if B > 5:
        areas_str += f"... ({B} blocks)"
    info = (
        f"block_consensus: B={B}, areas=[{areas_str}], "
        f"f̂_med={f_hat:.4f}, pct={pct:.1f}, thr={thr:.4f}, coverage={coverage:.1f}%"
    )
    return thr, filtered_attn, info


# =====================================================================
#  方法 12：Kneedle / Elbow Detection（排序曲線最大離差點）
# =====================================================================
def threshold_kneedle(
    filtered_attn: np.ndarray,
    low_attn: bool = False,
    **_kwargs,
) -> Tuple[float, np.ndarray, str]:
    """
    將 attention 值降序排列，正規化為 [0,1] 曲線，
    找到距離對角線 (0,1)→(1,0) 最遠的「肘部」位置作為閾值。

    x_i = i/N,  y_i = (a_i - a_N) / (a_1 - a_N)
    k* = argmax |y_i - (1 - x_i)|
    f_hat = k* / N
    """
    vals = np.sort(filtered_attn.ravel())[::-1]  # 降序
    N = len(vals)

    if N < 2 or vals[0] - vals[-1] < 1e-10:
        # 全部相同值
        pct = 75.0
        f_hat = 0.25
        thr = float(np.percentile(filtered_attn, pct))
        info = f"kneedle: uniform attn, fallback pct={pct:.0f}"
        if low_attn:
            thr = float(np.percentile(filtered_attn, 100.0 - pct))
            coverage = float((filtered_attn < thr).mean()) * 100
        else:
            coverage = float((filtered_attn >= thr).mean()) * 100
        info += f", thr={thr:.4f}, coverage={coverage:.1f}%"
        return thr, filtered_attn, info

    # 正規化
    x = np.arange(N, dtype=np.float64) / N
    y = (vals - vals[-1]) / (vals[0] - vals[-1])

    # 距離對角線 y = 1 - x 的偏差
    deviation = np.abs(y - (1.0 - x))
    k_star = int(np.argmax(deviation))

    f_hat = float(k_star) / N
    pct = float(np.clip((1.0 - f_hat) * 100.0, 50.0, 98.0))
    thr_val = float(vals[k_star])

    if low_attn:
        # 對 low_attn 仍用 percentile 確保一致性
        thr = float(np.percentile(filtered_attn, 100.0 - pct))
        coverage = float((filtered_attn < thr).mean()) * 100
    else:
        thr = thr_val
        coverage = float((filtered_attn >= thr).mean()) * 100

    info = (
        f"kneedle: k*={k_star}/{N}, f̂={f_hat:.4f}, pct={pct:.1f}, "
        f"thr={thr:.4f}, coverage={coverage:.1f}%"
    )
    return thr, filtered_attn, info


# =====================================================================
#  統一入口
# =====================================================================
def compute_threshold(
    filtered_attn: np.ndarray,
    method: int = 1,
    low_attn: bool = False,
    percentile: float = 75.0,
    ref_mask: Optional[np.ndarray] = None,
    max_iters: int = 20,
    fallback_percentile: float = 80.0,
    source_image: Optional[np.ndarray] = None,
    sigma_f: Optional[float] = None,
    percentile_min: float = 60.0,
    percentile_max: float = 90.0,
    cutoff_ratio: float = 0.25,
    R_min: float = 0.3,
    shrink_gamma: float = 1.0,
    attn_stack: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, str]:
    """
    統一閾值計算入口。

    Args:
        filtered_attn: (H, W) IQR 過濾後的 attention map（float32）
        method: 1~12，對應 12 種閾值策略
        low_attn: True = preserve mask（閾值以下為 True）
        percentile: 方法 1 的固定百分位數
        ref_mask: 方法 2 需要的 reference mask
        max_iters: 方法 2 的 ternary search 迭代次數
        fallback_percentile: 方法 2/8 的 fallback percentile
        source_image: 方法 6/8 需要的 source image (H, W, 3) uint8/float
        sigma_f: 方法 4 的 FFT sigma（None = 自動）
        percentile_min: 方法 5 的最低百分位
        percentile_max: 方法 5 的最高百分位
        cutoff_ratio: 方法 5/8 的頻率截止比
        R_min: 方法 8 的最低頻譜能量比
        shrink_gamma: 方法 9 的 IPR 收縮係數（預設 1.0）
        attn_stack: (B, H, W) 方法 11 需要的 IQR 過濾後各 block attention maps

    Returns:
        (threshold, processed_attn, info_str)
        - threshold: 閾值 float
        - processed_attn: 處理後的 attention map（方法 4/6/8 可能經過去噪）
        - info_str: 描述字串（用於 print）
    """
    if method == METHOD_FIXED_PERCENTILE:
        return threshold_fixed_percentile(filtered_attn, low_attn, percentile)

    elif method == METHOD_DYNAMIC_TERNARY:
        return threshold_dynamic_ternary(
            filtered_attn, low_attn, ref_mask, max_iters, fallback_percentile
        )

    elif method == METHOD_OTSU:
        return threshold_otsu(filtered_attn, low_attn)

    elif method == METHOD_FFT_OTSU:
        return threshold_fft_otsu(filtered_attn, low_attn, sigma_f)

    elif method == METHOD_SPECTRAL_ENERGY:
        return threshold_spectral_energy(
            filtered_attn, low_attn, percentile_min, percentile_max, cutoff_ratio
        )

    elif method == METHOD_EDGE_COHERENCE:
        return threshold_edge_coherence(filtered_attn, low_attn, source_image)

    elif method == METHOD_GMM:
        return threshold_gmm(filtered_attn, low_attn)

    elif method == METHOD_COMPOSITE:
        return threshold_composite(
            filtered_attn, low_attn, source_image,
            R_min, fallback_percentile, cutoff_ratio
        )

    elif method == METHOD_IPR:
        return threshold_ipr(filtered_attn, low_attn, shrink_gamma)

    elif method == METHOD_ENTROPY:
        return threshold_entropy(filtered_attn, low_attn)

    elif method == METHOD_BLOCK_CONSENSUS:
        return threshold_block_consensus(filtered_attn, low_attn, attn_stack)

    elif method == METHOD_KNEEDLE:
        return threshold_kneedle(filtered_attn, low_attn)

    else:
        raise ValueError(
            f"未知的 threshold method: {method}。"
            f"有效範圍：1~12，對應 {METHOD_NAMES}"
        )
