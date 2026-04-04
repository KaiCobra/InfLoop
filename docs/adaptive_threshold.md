# 自適應閾值方法文件（Adaptive Threshold Methods）

本文件說明 `infinity/utils/adaptiveThreshold.py` 中實作的 **8 種閾值策略**，用於將 IQR 過濾後的 cross-attention map 二值化為 focus / preserve mask。

---

## 前置處理：IQR 過濾（所有方法共用）

在進入任何閾值方法之前，各 transformer block 的 attention map 已經過 **IQR（Interquartile Range）離群值過濾**：

1. 計算各 block attention map 與全局平均的 MSE
2. 計算 MSE 的 Q1、Q3，得 $\text{IQR} = Q_3 - Q_1$
3. 移除 MSE 超過 $Q_3 + 1.5 \cdot \text{IQR}$ 的離群 block
4. 對剩餘 block 取平均，得到 $\tilde{A}_k(i,j)$ — 即 `filtered_attn`

所有 8 種方法均以 $\tilde{A}_k \in \mathbb{R}^{H \times W}$ 作為輸入。

---

## 統一介面

```python
from infinity.utils.adaptiveThreshold import compute_threshold

threshold, processed_attn, info_str = compute_threshold(
    filtered_attn,        # (H, W) np.ndarray float32
    method=3,             # 1~8
    low_attn=False,       # True = preserve mask（取低 attention）
    percentile=75.0,      # 方法 1 使用
    ref_mask=None,        # 方法 2 使用
    source_image=None,    # 方法 6/8 使用
    ...
)
```

**返回值**：
- `threshold`：浮點數閾值
- `processed_attn`：處理後的 attention map（方法 4/6/8 可能經過去噪，其餘方法原樣返回）
- `info_str`：描述字串（用於 logging）

**二值化規則**：
- `low_attn=False`（focus mask）：$\text{mask}(i,j) = [\hat{A}_k(i,j) \geq \tau]$
- `low_attn=True`（preserve mask）：$\text{mask}(i,j) = [\hat{A}_k(i,j) < \tau]$

---

## 方法 1：Fixed Percentile（固定百分位數）

**`--threshold_method 1`**

### 概念

最簡單的基準方法。直接取 attention map 值分佈的第 $p$ 百分位數作為閾值。

### 數學公式

$$
\tau = \text{Percentile}(\tilde{A}_k,\; p)
$$

其中 $p$ 由 `--attn_threshold_percentile`（預設 75）控制。

對於 `low_attn=True`（preserve mask），使用 $100 - p$ 百分位：

$$
\tau_{\text{low}} = \text{Percentile}(\tilde{A}_k,\; 100 - p)
$$

### 超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `percentile` | 75.0 | 閾值百分位數 |

### 特性

- ✅ 無需額外輸入，計算最快
- ❌ 純 rank-based，忽略分佈形狀
- ❌ 不同 scale、不同圖像都用同一個 percentile，缺乏自適應性

---

## 方法 2：Dynamic Ternary Search（動態三分搜尋）

**`--threshold_method 2`**

### 概念

利用 PIE-Bench 提供的 ground-truth reference mask，透過三分搜尋找到使 IoU 最大化的百分位閾值。屬於 **oracle-guided** 方法，用於實驗上界估計。

### 數學公式

**目標函式**：在百分位空間 $p \in [0, 100]$ 上最大化 IoU：

$$
p^* = \arg\max_{p \in [0, 100]} \; \text{IoU}\bigl(\mathcal{M}_p,\; \mathcal{M}_{\text{ref}}\bigr)
$$

其中候選 mask 為：

$$
\mathcal{M}_p(i,j) = \bigl[\tilde{A}_k^{\uparrow}(i,j) \geq \text{Percentile}(\tilde{A}_k^{\uparrow}, p)\bigr]
$$

$\tilde{A}_k^{\uparrow}$ 為將 attention resize 到 reference mask 解析度（512×512）後的結果。

**IoU 計算**：

$$
\text{IoU}(\mathcal{M}_p, \mathcal{M}_{\text{ref}}) = \frac{|\mathcal{M}_p \cap \mathcal{M}_{\text{ref}}|}{|\mathcal{M}_p \cup \mathcal{M}_{\text{ref}}|}
$$

**三分搜尋**（$T$ 次迭代）：

$$
\begin{cases}
m_1 = l + \frac{h - l}{3}, \quad m_2 = h - \frac{h - l}{3} \\[6pt]
\text{若 } \text{IoU}(m_1) < \text{IoU}(m_2): & l \leftarrow m_1 \\
\text{否則}: & h \leftarrow m_2
\end{cases}
$$

最終 $p^* = \frac{l + h}{2}$，對應閾值 $\tau = \text{Percentile}(\tilde{A}_k, p^*)$。

### 超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `ref_mask` | — | (H_ref, W_ref) bool reference mask（必須提供） |
| `max_iters` | 20 | 三分搜尋迭代次數 |
| `fallback_percentile` | 80.0 | ref_mask 不可用時的 fallback percentile |

### Fallback 條件

- `ref_mask` 為 None、全零或全一 → fallback 到方法 1
- 最佳 IoU < 0.01 或 coverage < 1% → fallback 到方法 1

### 特性

- ✅ 理論上界：有 GT mask 時可找到最佳閾值
- ❌ 需要 ground-truth mask，不適用於實際推論
- ⚠️ 僅適合 PIE-Bench 實驗與 ablation

---

## 方法 3：Otsu 最大類間方差法

**`--threshold_method 3`**

### 概念

假設 attention 值的直方圖呈雙模態（背景 + 前景），Otsu 方法自動找到使**類間方差最大化**的閾值，無需手動設定百分位。

### 數學公式

將 $\tilde{A}_k$ 的值域正規化到 $[0, 255]$（uint8）：

$$
A_{\text{norm}}(i,j) = \left\lfloor \frac{\tilde{A}_k(i,j) - A_{\min}}{A_{\max} - A_{\min}} \times 255 \right\rfloor
$$

Otsu 目標：最大化類間方差 $\sigma_B^2$：

$$
\tau_{\text{otsu}} = \arg\max_{t \in [0, 255]} \; \sigma_B^2(t)
$$

$$
\sigma_B^2(t) = w_0(t) \cdot w_1(t) \cdot \bigl(\mu_0(t) - \mu_1(t)\bigr)^2
$$

其中：
- $w_0(t) = \sum_{i=0}^{t} p_i$：背景像素比例
- $w_1(t) = 1 - w_0(t)$：前景像素比例
- $\mu_0(t), \mu_1(t)$：兩類的平均灰度值

**反映射**回原始值域：

$$
\tau = A_{\min} + \frac{\tau_{\text{otsu}}}{255} \cdot (A_{\max} - A_{\min})
$$

### 超參數

無。完全自動。

### 特性

- ✅ 無超參數，完全自適應
- ✅ 計算高效（O(N) histogram sweep）
- ❌ 假設雙模態分佈；若 attention 分佈不符合，閾值可能不理想
- ❌ 未考慮空間結構，純基於值域統計

---

## 方法 4：FFT Low-Pass + Otsu（FFT 低通去噪 + Otsu）

**`--threshold_method 4`**

### 概念

Attention map 在 coarse scales 易受 block 噪聲干擾。先用 **2D FFT 高斯低通濾波**去除高頻噪聲，再對平滑後的 attention 套用 Otsu 閾值。

### 數學公式

**Step 1：FFT 高斯低通濾波**

$$
F_A = \text{FFT2}(\tilde{A}_k)
$$

高斯低通遮罩：

$$
G(\omega_y, \omega_x) = \exp\!\Biggl(-\frac{\omega_y^2 + \omega_x^2}{2\sigma_f^2}\Biggr)
$$

$$
\hat{A}_k = \text{IFFT2}\bigl(F_A \cdot G\bigr)
$$

其中 $\sigma_f$ 為頻率域高斯 sigma；預設 $\sigma_f = \max(H, W) / 4$。

**Step 2：Otsu 閾值**

對去噪後的 $\hat{A}_k$ 套用 Otsu（同方法 3）。

### 超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `sigma_f` | `max(H,W)/4` | 頻率域高斯 sigma（None = 自動） |

### Fallback 條件

- $H < 3$ 或 $W < 3$ → 直接 fallback 到 Otsu（方法 3）

### 特性

- ✅ 有效抑制高頻雜訊，提升 Otsu 穩定性
- ✅ 返回去噪後的 attention（可用於視覺化）
- ❌ 可能過度平滑，導致小的 focus 區域被模糊掉
- ⚠️ $\sigma_f$ 影響保留的空間細節程度

---

## 方法 5：Spectral Energy Ratio（頻譜能量比自適應閾值）

**`--threshold_method 5`**

### 概念

利用 attention map 的 **低頻能量佔比** $R_k$ 衡量其空間集中度。$R_k$ 大代表 attention 清晰集中（blob 形），可用較 selective 的閾值；$R_k$ 小代表分散噪聲，需更保守的閾值。

### 數學公式

**Step 1：計算低頻能量比 $R_k$**

$$
F_A = \text{FFT2}(\tilde{A}_k), \quad P(\omega) = |F_A(\omega)|^2
$$

定義低頻區域（截止頻率 $\omega_c = \text{cutoff\_ratio} \times f_{\text{Nyquist}}$）：

$$
\Omega_{\text{low}} = \left\{(\omega_y, \omega_x) \;\middle|\; \left(\frac{\omega_y}{\omega_{c,y}}\right)^2 + \left(\frac{\omega_x}{\omega_{c,x}}\right)^2 \leq 1 \right\}
$$

$$
R_k = \frac{\sum_{\omega \in \Omega_{\text{low}}} P(\omega)}{\sum_{\omega} P(\omega)} \in [0, 1]
$$

**Step 2：線性映射到百分位**

$$
p = p_{\max} - R_k \cdot (p_{\max} - p_{\min})
$$

- $R_k \to 1$（清晰 blob）→ $p \to p_{\min}$（更 selective，取更少像素為 focus）
- $R_k \to 0$（分散噪聲）→ $p \to p_{\max}$（更保守，取更多像素為 focus）

$$
\tau = \text{Percentile}(\tilde{A}_k,\; p)
$$

### 超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `percentile_min` | 60.0 | $R_k=1$ 時的目標百分位（最 selective） |
| `percentile_max` | 90.0 | $R_k=0$ 時的目標百分位（最保守） |
| `cutoff_ratio` | 0.25 | 低頻截止比（佔 Nyquist 頻率的比例） |

### Fallback 條件

- $H < 3$ 或 $W < 3$ → 使用 $p = (p_{\min} + p_{\max})/2$

### 特性

- ✅ 根據 attention 的空間集中度自動調整閾值
- ✅ 不需要 reference mask 或 source image
- ❌ 仍屬 percentile-based，只是動態調整百分位
- ⚠️ $R_k$ 到 percentile 的映射為線性假設

---

## 方法 6：Edge-Attention Cross-Spectral Coherence（跨頻譜相干性）

**`--threshold_method 6`**

### 概念

利用 **source image 的邊緣資訊** 作為空間先驗：物件邊界處的 attention 更可能是真正的 focus 信號，而非噪聲。透過頻率域的 **相位相干性** (phase coherence) 濾波，只保留與 image edge 同相的 attention 頻率成分。

### 數學公式

**Step 1：計算 Source Image Edge Map**

$$
E = \sqrt{(S_x * I)^2 + (S_y * I)^2}
$$

其中 $S_x, S_y$ 為 Sobel kernel，$I$ 為灰階 source image。$E$ 被 resize 到 attention 解析度 $(H, W)$。

**Step 2：跨頻譜相位相干性濾波**

$$
F_A = \text{FFT2}(\tilde{A}_k), \quad F_E = \text{FFT2}(E)
$$

相干性係數：

$$
\gamma(\omega) = \frac{\text{Re}\bigl(F_A(\omega) \cdot F_E^*(\omega)\bigr)}{|F_A(\omega)| \cdot |F_E(\omega)| + \varepsilon}
\quad \in [-1, 1]
$$

只保留正相干成分：

$$
\tilde{F}_A(\omega) = F_A(\omega) \cdot \max\bigl(0,\; \gamma(\omega)\bigr)
$$

$$
\hat{A}_k = \text{IFFT2}(\tilde{F}_A)
$$

$\hat{A}_k$ 中僅保留 「與邊緣同相」 的 attention 成分，噪聲及非結構信號被抑制。

**Step 3：Otsu 閾值**

對去噪後的 $\hat{A}_k$ 套用 Otsu（同方法 3）。

### 超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `source_image` | — | (H, W, 3) uint8 source image（必須提供） |
| $\varepsilon$ | $10^{-8}$ | 數值穩定項 |

### Fallback 條件

- 無 `source_image` → fallback 到 Otsu（方法 3）
- $H < 3$ 或 $W < 3$ → fallback 到 Otsu
- Edge map 全零（純色圖片）→ fallback 到 Otsu
- 濾波後 $\hat{A}_k$ 退化（全同值）→ fallback 到 Otsu

### 特性

- ✅ 有效利用 source image 的空間結構資訊
- ✅ 物件邊界處的 attention 被增強，內部噪聲被抑制
- ❌ 需要 source image
- ❌ 若 source image 邊緣與 attention focus 不對齊（例如新增物件），效果有限
- ⚠️ 返回去噪後的 $\hat{A}_k$，閾值以此為基準

---

## 方法 7：GMM 雙高斯混合模型

**`--threshold_method 7`**

### 概念

假設 attention 分佈由兩個高斯（**背景** $\mathcal{N}(\mu_0, \sigma_0^2)$ 和 **前景** $\mathcal{N}(\mu_1, \sigma_1^2)$，$\mu_0 < \mu_1$）混合而成。用 EM 演算法 fit 後，取兩個分佈的**決策邊界**（等密度點）作為閾值。

### 數學公式

**混合模型**：

$$
p(a) = \pi_0 \;\mathcal{N}(a \mid \mu_0, \sigma_0^2) + \pi_1 \;\mathcal{N}(a \mid \mu_1, \sigma_1^2)
$$

其中 $\pi_0 + \pi_1 = 1$ 為混合權重。

**EM 演算法**：

初始化：用中位數將資料分為兩半，分別估計初始 $\mu_0, \sigma_0, \mu_1, \sigma_1$。

**E-step**（計算 responsibilities）：

$$
r_{n,0} = \frac{\pi_0 \;\mathcal{N}(a_n \mid \mu_0, \sigma_0^2)}{\pi_0 \;\mathcal{N}(a_n \mid \mu_0, \sigma_0^2) + \pi_1 \;\mathcal{N}(a_n \mid \mu_1, \sigma_1^2)}
$$

$$
r_{n,1} = 1 - r_{n,0}
$$

**M-step**（更新參數）：

$$
N_j = \sum_{n} r_{n,j}, \quad
\pi_j = \frac{N_j}{N}, \quad
\mu_j = \frac{1}{N_j}\sum_{n} r_{n,j} a_n, \quad
\sigma_j = \sqrt{\frac{1}{N_j}\sum_{n} r_{n,j}(a_n - \mu_j)^2}
$$

迭代至 log-likelihood 收斂（$|\Delta \mathcal{L}| < 10^{-6}$）或達到最大迭代次數 50。

**決策邊界**：

在 $[\mu_0, \mu_1]$ 之間找到兩個 component 密度相等的交叉點：

$$
\tau = \arg_{a \in [\mu_0, \mu_1]} \; \pi_0 \;\mathcal{N}(a \mid \mu_0, \sigma_0^2) = \pi_1 \;\mathcal{N}(a \mid \mu_1, \sigma_1^2)
$$

使用數值線性搜尋（200 個採樣點 + 線性內插）求解。

### 超參數

無外部超參數。EM 內部：
- 最大迭代 50 次
- log-likelihood 收斂閾值 $10^{-6}$

### Fallback 條件

- $|\mu_1 - \mu_0| < 10^{-8}$（退化：兩 component 重合）→ fallback 到 Otsu
- 決策邊界計算失敗 → 使用 $(\mu_0 + \mu_1)/2$

### 特性

- ✅ 明確的概率模型，比 Otsu 更靈活（允許不等方差、不等權重）
- ✅ 無外部超參數
- ❌ 假設雙模態高斯分佈；若分佈更複雜（多峰或長尾），模型不足
- ❌ EM 可能收斂到局部最優
- ❌ 依賴 `scipy.stats.norm`（額外依賴）

---

## 方法 8：Composite（複合方案）

**`--threshold_method 8`**

### 概念

結合方法 5（$R_k$ 信心度量）、方法 6（Edge-Coherent 去噪）、方法 3（Otsu 閾值）的複合策略，並根據頻譜能量比自動判斷是否需要 fallback。

### 流程圖

```
source_image available?
├── YES → Edge-Coherent Filter (同方法 6)
│          → ã = IFFT2(F_A · max(0, γ))
│          edge_map flat? → YES → FFT Low-Pass
└── NO  → FFT Low-Pass (σ_f = max(H,W)/4)

         ↓ ã (denoised attention)

計算 R_k = 低頻能量比 (同方法 5)
         ↓
R_k ≥ R_min?
├── YES → Otsu(ã)
│          → τ = Otsu threshold on denoised attention
└── NO  → Fallback to Fixed Percentile
           → τ = Percentile(Ã_k, p_fallback)
```

### 數學公式

**Step 1：去噪**

若有 source image：
$$
\hat{A}_k = \text{IFFT2}\bigl(F_A \cdot \max(0, \gamma)\bigr) \quad \text{（Edge-Coherent，同方法 6）}
$$

若無 source image：
$$
\hat{A}_k = \text{IFFT2}\bigl(F_A \cdot G \bigr) \quad \text{（FFT Low-Pass，同方法 4）}
$$

**Step 2：信心度量**

$$
R_k = \frac{\sum_{\omega \in \Omega_{\text{low}}} |\hat{F}_A(\omega)|^2}{\sum_{\omega} |\hat{F}_A(\omega)|^2}
$$

**Step 3：閾值決策**

$$
\tau = 
\begin{cases}
\text{Otsu}(\hat{A}_k) & \text{if } R_k \geq R_{\min} \\[4pt]
\text{Percentile}(\tilde{A}_k,\; p_{\text{fallback}}) & \text{if } R_k < R_{\min}
\end{cases}
$$

### 超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `source_image` | None | source image（可選；無時用 FFT 低通） |
| `R_min` | 0.3 | 頻譜能量比最低門檻 |
| `fallback_percentile` | 75.0 | $R_k < R_{\min}$ 時的 fallback percentile |
| `cutoff_ratio` | 0.25 | $R_k$ 計算的頻率截止比 |

### 特性

- ✅ 最穩健的方法：有 source image 時利用邊緣資訊，無時自動降級
- ✅ $R_k$ 信心機制避免在噪聲 attention 上強行 Otsu
- ✅ 適應各種場景：有/無 source image、清晰/模糊 attention
- ❌ 當 $R_k < R_{\min}$ 時退化成固定 percentile
- ⚠️ 返回去噪後的 attention，與原始 attention 值域可能不同

---

## 方法比較總覽

| # | 方法 | 需 source image | 需 ref mask | 超參數 | 自適應 | 備註 |
|---|------|:-:|:-:|:---:|:-:|------|
| 1 | Fixed Percentile | ✗ | ✗ | `percentile` | ✗ | 基準方法 |
| 2 | Dynamic Ternary | ✗ | ✔ | `max_iters` | ✔ | Oracle 上界，僅 PIE-Bench |
| 3 | Otsu | ✗ | ✗ | 無 | ✔ | 最大類間方差，全自動 |
| 4 | FFT + Otsu | ✗ | ✗ | `sigma_f` | ✔ | FFT 去噪 + Otsu |
| 5 | Spectral Energy | ✗ | ✗ | `percentile_min/max`, `cutoff_ratio` | ✔ | $R_k$ 自適應百分位 |
| 6 | Edge Coherence | ✔ | ✗ | 無 | ✔ | 頻率域相位濾波 |
| 7 | GMM | ✗ | ✗ | 無 | ✔ | EM 雙高斯混合模型 |
| 8 | Composite | 可選 | ✗ | `R_min`, `cutoff_ratio` | ✔ | 6→3→5 複合策略 |

---

## Shell Script 使用方式

```bash
# infer_p2p_edit.sh 或 batch_run_pie_edit.sh
threshold_method=3   # 改這裡即可切換
```

或直接傳 CLI 參數：

```bash
python3 tools/run_p2p_edit.py \
  --threshold_method 3 \
  ...

python3 tools/batch_run_pie_edit.py \
  --threshold_method 3 \
  ...
```

批量跑所有方法：

```bash
for m in 1 2 3 4 5 6 7 8; do
  sed -i "s/^threshold_method=.*/threshold_method=$m/" scripts/batch_run_pie_edit.sh
  bash scripts/batch_run_pie_edit.sh
done
```

---

## 原始碼位置

- 閾值模組：`infinity/utils/adaptiveThreshold.py`
- 呼叫入口：`tools/run_p2p_edit.py` → `compute_attention_mask_for_scale()`
- PIE-Bench 路徑：`tools/run_pie_edit.py` → `collect_attention_text_masks_dynamic()`
