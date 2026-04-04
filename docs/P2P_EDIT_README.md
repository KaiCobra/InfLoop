# P2P-Edit 管線（Source Image Injection + Attention-Guided P2P）

## 概述

P2P-Edit 是在 P2P-Attn 管線基礎上的進化版本，透過 **source image 直接 VAE 編碼注入** 與 **雙路 attention 空間遮罩**，實現在保持 source image 結構的同時，精確替換局部語義內容。

### 近期更新（與目前程式一致）

- 新增 `--auto_focus_from_prompt_diff`（預設 `1`）：自動把 source/target prompt 詞級差異加入各自 focus terms。
- 新增 `--use_cumulative_prob_mask`（預設 `0`）：可將 replacement mask 轉為跨尺度累積機率遮罩。
- Phase 1.7 preserve 錨定僅在「有 source image token + 有 source low-attn mask」時啟用，否則自動退回 free-gen。
- 當 `source_focus_words` 與 `target_focus_words` 僅一側非空時，啟用 **single-focus fallback**：跳過 Phase 1.7 guided gen，改為 **source image 連續特徵注入**前 N scale（`--phase17_fallback_replace_scales`，預設 4），再自由生成剩餘 scale，從中擷取 target attention。
- 當提供的 PIE mask 為全白（100% white）時，會忽略 PIE mask，改用 source/target prompt attention mask。
- 新增 `--phase17_fallback_replace_scales`（預設 `4`）：single-focus fallback 時，Phase 1.7 以 source image 連續特徵注入前幾個 scale 作為結構錨定（0=停用，純 free-gen）。
- 新增 `--debug_mode`（預設 `0`）：設為 1 時儲存所有中間階段圖片（`phase17_guided.jpg`、`phase17_fallback.jpg`），方便除錯與 ablation。
- `run_pie_edit.py` 支援**單一案例模式**：不需 `--bench_dir`，直接傳入 `--source_image` + `--source_prompt` + `--target_prompt` 即可執行單張圖片編輯。
- 新增 `scripts/single_infer_pie_edit.sh`：PIE-Bench 單一案例快速執行腳本，自動從 `meta.json` 讀取 prompt 與 focus words。
- PIE mask 相關參數：`--use_pie_mask`（0/1/2）、`--pie_mask_attn_fallback`、`--mask_expand_percent`。
- Attention cache（Prompt-to-Prompt 風格）：`--use_attn_cache`、`--attn_cache_phase`、`--attn_cache_max_scale`、`--attn_cache_align_mode`（blended / full_p2p）。

### 與 P2P-Attn 的核心差異

| 特性 | P2P-Attn | P2P-Edit |
|------|----------|----------|
| Token 替換來源 | Source gen 採樣 token（受 source prompt 語義影響）| Source image 直接 VAE 量化 token（純像素）|
| Source 生成引導 | 純文字（source prompt 驅動）| Source image 連續特徵注入 summed_codes |
| Focus mask 來源 | 僅 source attention | Source + Target 雙路 attention union |
| Source image 輸入 | 不需要 | 可選（留空 → 自動退回 P2P-Attn 模式）|

### 使用場景

```
Ps: a dog wearing space suit
Pt: a dog
```

**結果**：
- 狗的外型、姿態、背景 → 與 source image 高度一致（pixel-level 結構保留）
- 太空裝區域 → 由 target prompt 自由生成，移除該物件

---

## 架構

```
InfLoop/
├── infinity/
│   ├── models/
│   │   └── infinity_p2p_edit.py          # P2P-Edit 版模型（支援連續特徵注入 + P2P token 替換）
│   └── utils/
│       └── bitwise_token_storage.py      # Token + 遮罩儲存（共用）
├── tools/
│   ├── run_p2p_edit.py                   # P2P-Edit 基礎引擎（Phase 0~2 + gen_one_img）
│   ├── run_pie_edit.py                   # PIE-Bench P2P-Edit 評估（單一/批量模式）
│   └── batch_run_pie_edit.py             # PIE-Bench 批量執行器（呼叫 run_pie_edit.run_one_case）
├── scripts/
│   ├── infer_p2p_edit.sh                 # 單一案例 P2P-Edit 執行腳本
│   ├── single_infer_pie_edit.sh          # 單一案例 PIE-Bench P2P-Edit（自動讀取 meta.json）
│   └── batch_run_pie_edit.sh             # PIE-Bench 批量執行腳本
├── attention_map/
│   └── extractor.py                      # CrossAttentionExtractor（現有，共用）
└── docs/
    ├── P2P_EDIT_README.md                # 本文件
    └── P2P_EDIT_QUICKSTART.md            # 快速上手
```

---

## 工作流程

### 七個階段

```
┌─────────────────────────────────────────────────────────────────┐
│         P2P-Edit 管線（七個階段）                                │
└─────────────────────────────────────────────────────────────────┘

┌── Phase 0：Source Image 編碼 ────────────────────────────────────┐
│                                                                 │
│  Source Image（PNG/JPG）                                        │
│       ↓                                                         │
│  (A) encode_image_to_raw_features()                            │
│      → 連續 VAE encoder 輸出 [1, d, 1, H, W]                   │
│      → 用於 Phase 1 source gen 的 summed_codes 注入             │
│                                                                 │
│  (B) encode_image_to_scale_tokens()                            │
│      → vae.encode() → all_bit_indices                          │
│      → 各 scale 的離散 bit token Dict[si → [1,1,h,w,d]]        │
│      → Phase 1.9 後覆寫 storage.tokens                         │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌── Phase 1：Source 生成 + Attention 擷取 ─────────────────────────┐
│                                                                 │
│  Source Prompt + image_raw_features                            │
│       ↓                                                         │
│  [CrossAttentionExtractor 掛載]                                │
│  Infinity 自回歸生成（前 N scale 注入 image raw features）      │
│  ├─ summed_codes = lerp(image_features, gen_codes, w_si)       │
│  ├─ 擷取各 scale 的 cross-attention map                        │
│  └─ 儲存所有 scale 的 bitwise token → BitwiseTokenStorage      │
│       ↓                                                         │
│  Source Image（重建）+ Token Storage + Extractor               │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌── Phase 1.5：Source Focus Mask 計算（雙向）──────────────────────┐
│                                                                 │
│  source_focus_words（如 "space suit"）+ 同一組 attention map    │
│       ↓ 同時計算兩種遮罩（不需額外 forward pass）               │
│                                                                 │
│  ① 高 attention focus mask（source_text_masks）                │
│     閾值 = attn_threshold_percentile（如 80）                   │
│     最高 (100-80)% = 20% 的注意力區域 → True（不替換）          │
│                                                                 │
│  ② 低 attention preserve mask（source_low_attn_masks）         │
│     閾值 = 100 - attn_threshold_percentile（如 20）             │
│     最低 20% 的注意力區域 → True（Phase 1.7 強制保留）           │
│     ★ 這是與 source image「最無關」的純背景區域                  │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌── Phase 1.6：建立 Phase 1.7 Preserve Storage ────────────────────┐
│                                                                 │
│  for each scale in source_low_attn_masks:                      │
│  ├─ phase17_storage.tokens[si] = image_scale_tokens[si]        │
│  │   （直接使用 source image VAE 量化 token）                   │
│  └─ phase17_storage.masks[si] = source_low_attn_mask           │
│      （True = 低 attention 背景 = Phase 1.7 強制用 source token）│
└─────────────────────────────────────────────────────────────────┘
         ↓
┌── Phase 1.7：Target 引導生成（擷取 Attention）─────────────────┐
│                                                                 │
│  【一般模式】（source + target 都有 focus words）               │
│  Target Prompt + phase17_storage（preserve mask）              │
│       ↓                                                         │
│  [Target CrossAttentionExtractor 掛載]                         │
│  Scale 0 ~ N-1：100% source image token（結構錨定）             │
│  Scale N+  ：                                                   │
│  ├─ 低 attention 背景區（preserve mask=True）                   │
│  │   → 強制使用 source image token（消除 source/model 衝突）    │
│  └─ 高 attention focus 區（preserve mask=False）                │
│      → target prompt 自由生成（反映 target 語義）               │
│  → debug_mode=1 時儲存 phase17_guided.jpg                      │
│                                                                 │
│  【Single-focus fallback】（僅 target focus，無 source focus）  │
│  Target Prompt + inject_image_features + inject_schedule       │
│       ↓                                                         │
│  [Target CrossAttentionExtractor 掛載]                         │
│  Scale 0 ~ M-1：source image 連續特徵注入（weight=0.0）        │
│                  M = phase17_fallback_replace_scales（預設 4）   │
│  Scale M+  ：target prompt 自由生成（weight=1.0）               │
│  → 結構來自 summed_codes 的 source image 特徵，非離散 token     │
│  → debug_mode=1 時儲存 phase17_fallback.jpg                    │
│                                                                 │
│  target_focus_words → target_text_masks                        │
│  （此步驟產生的圖像不儲存，僅用於收集 attention）               │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌── Phase 1.9：合併 Mask → 覆寫 storage.tokens ──────────────────┐
│                                                                 │
│  union(source_text_masks, target_text_masks)                   │
│       ↓                                                         │
│  replacement_mask = ~union（True = 背景，False = focus 區域）   │
│  → 存入 p2p_token_storage.masks[si]                            │
│                                                                 │
│  ★ P2P-Edit 特有：覆寫 storage.tokens                          │
│  for si in image_scale_tokens:                                 │
│      storage.tokens[si] = image_scale_tokens[si]               │
│  → Phase 2 的 P2P 替換現在參考 source image 像素，              │
│    而非 source gen 採樣（消除 source prompt 的語義偏差）        │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌── Phase 2：Target 生成（純 P2P-Attn 模式）─────────────────────┐
│                                                                 │
│  Target Prompt                                                  │
│       ↓                                                         │
│  Scale 0 ~ N-1（全域替換，離散 token 層級）：                   │
│      100% 替換為 storage.tokens[si]                            │
│      （= source image 直接量化 token，保留整體結構）            │
│                                                                 │
│  Scale N ~ end（Attention 遮罩引導）：                          │
│      ├─ 背景區域（replacement_mask=True）→ 替換 source token    │
│      │   （保留背景細節）                                       │
│      └─ Focus 區域（replacement_mask=False）→ 保留 target 自由生成
│          （允許 target prompt 改變此區域）                       │
│                                                                 │
│  ⚠ 注意：Phase 2 永遠不注入 source image（inject_image_features=None）
│    目的：確保 target 可根據 target prompt 自由改變語義內容     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 快速開始

### 方式一：單一案例 P2P-Edit

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/infer_p2p_edit.sh
```

### 方式二：單一案例 PIE-Bench P2P-Edit（自動從 meta.json 讀取 prompt）

```bash
cd /home/avlab/Documents/InfLoop
# 修改 scripts/single_infer_pie_edit.sh 中的 dir= 路徑
bash scripts/single_infer_pie_edit.sh
```

### 方式三：PIE-Bench 批量評估

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/batch_run_pie_edit.sh
```

結果儲存於 `./outputs/p2p_edit_YYYYMMDD_HHMMSS/`：
- `source.jpg` — Source 重建圖像（含 image injection）
- `target.jpg` — 編輯後圖像（P2P-Attn 引導）
- `phase17_guided.jpg` — Phase 1.7 引導生成圖像（`--debug_mode 1` 時）
- `phase17_fallback.jpg` — Phase 1.7 fallback 生成圖像（`--debug_mode 1`，single-focus 時）
- `attn_masks/source/` — Source focus 遮罩（黑色=focus 不替換，白色=背景替換）
- `attn_masks/phase17_preserve/` — Phase 1.7 低 attention preserve 遮罩（**白色=錨定區域**）★
- `attn_masks/target/` — Target focus 遮罩（黑色=focus 不替換，白色=背景替換）
- `attn_masks/combined/` — 合併 replacement 遮罩（**白色=source token 替換，黑色=union 保護區**）

---

## 參數說明

### 核心 Prompt 參數

| 參數 | 說明 |
|------|------|
| `--source_prompt` | Source prompt（含欲修改的物件/詞彙）|
| `--target_prompt` | Target prompt（修改後的內容）|
| `--source_focus_words` | Source prompt 中欲關注的詞彙（空格分隔）|
| `--target_focus_words` | Target prompt 中對應的新詞彙（可為空字串）|
| `--auto_focus_from_prompt_diff` | 是否自動加入 source/target prompt 差異詞彙（`1`=啟用，預設 `1`）|

### Source Image 注入參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--source_image` | `None`（可選）| Source image 路徑。留空 → 純 P2P-Attn 模式 |
| `--image_injection_scales` | `2` | 前幾個 scale 使用 source image 注入（weight=0 = 100% image）|
| `--inject_weights` | 自動 | 各 scale 的注入強度，空格分隔。0.0=100% source，1.0=自由生成。<br>若指定，覆蓋 `image_injection_scales` |

**inject_weights 範例**（pn=1M，共 13 個 scale）：
```
# 目前設定：全部 scale 完全注入 source image（結構保留最強）
"0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"

# 部分注入範例：前 2 scale 完全注入，其餘自由生成
"0.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0"
 ^^^  ^^^  ─────────────────── 以下完全自由生成 ──────────
 scale0  scale1 (完全注入 source image)
```

### P2P Token 替換參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--num_full_replace_scales` | `2` | 前幾個 scale 做 100% source token 替換（建議與 `image_injection_scales` 一致）|
| `--attn_threshold_percentile` | `80`（PIE）/ `75`（P2P-Edit CLI） | 高於此百分位 = focus 區域（不替換）。值越小 = focus 區越大（例如 80 = 最高 20% 為 focus；最低 20% 為 Phase 1.7 preserve）|
| `--p2p_token_replace_prob` | `0.0` | Fallback 機率替換（有遮罩才生效）；0.0 表示不啟用 |
| `--use_cumulative_prob_mask` | `0` | `1`=跨尺度累積機率遮罩（灰階機率替換）；`0`=單一 bool 遮罩 |
| `--save_attn_vis` | `1`（P2P-Edit）/ `0`（PIE 批量） | 是否儲存遮罩視覺化 |

### Phase 1.7 Fallback 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--phase17_fallback_replace_scales` | `4` | single-focus fallback 時，Phase 1.7 以 source image **連續特徵**注入前幾個 scale（0=停用，純 free-gen）|

> **設計原因**：add_object 等任務中 `source_focus_words=""`（無 source focus），但有 `target_focus_words`。這時 Phase 1.7 若純自由生成，產出的圖像與 source image 結構毫無關聯，導致 target attention map 無法反映真正需要編輯的空間位置。改用 source image 連續特徵注入前 N scale，Phase 1.7 的 summed_codes 在粗略 scale 錨定了 source 結構，target prompt 在細節 scale 仍能自由生成新增物件。

### Debug 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--debug_mode` | `0` | `1`=儲存所有中間階段圖片（`phase17_guided.jpg`、`phase17_fallback.jpg`），方便除錯與 ablation |

### Attention Block 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--attn_block_start` | `-1` | 起始 block index（-1 = 自動使用後半段 block；可手動設為 2）|
| `--attn_block_end` | `-1` | 結束 block index（-1 = 自動，使用最後一個 block）|
| `--attn_batch_idx` | `0` | CFG 下擷取哪個 batch（0 = conditioned，對應 source/target prompt）|

> `scripts/infer_p2p_edit.sh` 常覆蓋 CLI 預設（例如 `attn_threshold_percentile=80`、`attn_block_start=2`）。

> **`attn_threshold_percentile` 同時控制兩個閾值**：
> - 高 attention focus（Phase 1.5 ①、Phase 2 不替換）：最高 `(100 - p)%`
> - 低 attention preserve（Phase 1.5 ②、Phase 1.7 錨定）：最低 `(100 - p)%`
> 兩者對稱，各佔相同比例（如 p=80 → 各 20%）。

### PIE Mask 參數（`run_pie_edit.py` 專用）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--use_pie_mask` | `0` | PIE mask 使用模式：`0`=關閉、`1`=直接作為 replacement mask、`2`=僅用白色比例反推 `attn_threshold_percentile`（65~92 反向線性）|
| `--pie_mask_attn_fallback` | `0` | （需 `use_pie_mask=1`）白色區域內以 attention mask 二次篩選「真正需要編輯」的 token |
| `--mask_expand_percent` | `0.0` | 每個 scale 對最終 replacement mask 向外擴張 True 區域的比例（% of min(H,W)）|

### Attention Cache 參數（Prompt-to-Prompt 風格，`run_pie_edit.py` 專用）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--use_attn_cache` | `0` | 是否啟用 blended_words 對齊的 attention cache/replace |
| `--attn_cache_phase` | `phase2` | 在哪個 phase 套用：`phase17`、`phase2`、`both` |
| `--attn_cache_max_scale` | `13` | 從 scale 1 開始，最多套用到第幾個 scale（1-based）|
| `--attn_cache_align_mode` | `blended` | `blended`=僅 blended_words 指定的 token 做替換；`full_p2p`=完整 Prompt-to-Prompt 對齊 |

---

## 技術細節

### 1. 兩種 Source Image 編碼路徑

P2P-Edit 對 source image 進行**兩種獨立編碼**，各司其職：

#### (A) `encode_image_to_raw_features()`
```
Source Image → vae.encode_for_raw_features() → [1, d, 1, H, W]
```
- **連續特徵**，不做量化 round-trip
- 用途：Phase 1 source gen 的 `summed_codes` 混合
- 效果：Source gen 的解碼結果貼近 source image 像素，增強結構保留

#### (B) `encode_image_to_scale_tokens()`
```
Source Image → vae.encode() → all_bit_indices → Dict[si → [1,1,h,w,d]]
```
- **離散 bit token**，與 transformer 自回歸生成的 `idx_Bld` 完全同格式
- 用途：Phase 1.9 後覆寫 `storage.tokens[si]`，供 Phase 2 P2P 替換使用
- 效果：Phase 2 替換的 token 直接來自 source image 量化，不受 source prompt 語義偏差影響

### 2. Token 覆寫的設計原因

```
P2P-Attn（原版）：
storage.tokens[si] = source gen 採樣的 idx_Bld
                     → 受 source prompt 語義影響，不一定 = source image 像素

P2P-Edit：
storage.tokens[si] = image_scale_tokens[si]（覆寫）
                     → source image 直接量化，精確反映像素內容
```

這項差異在 source prompt 與 source image 不完全對應時（如 source image 是真實照片
而非由 prompt 生成）效果尤為顯著。

### 3. Mask 視覺化規則

| 遮罩圖 | 白色 | 黑色 |
|--------|------|------|
| Source focus（`invert=True`）| 背景替換區 | Focus 區（target 自由生成）|
| Phase 1.7 preserve（`invert=False`）| **低 attention 錨定區**（source token 強制保留）| Focus 自由區（preserve mask=False）|
| Target focus（`invert=True`）| 背景替換區 | Focus 區（改變）|
| Combined replacement（`invert=True`）| `replacement_mask=True`（背景，替換）| Union focus 區（保護，不替換）|

### 4. Phase 2 不注入 source image 的原因

Phase 2 target gen 永遠不使用 `inject_image_features`（固定傳入 `None`），原因是：

```
若 Phase 2 也注入 source image：
  target prompt 在被注入的 scale 根本無法改變任何內容
  → target 幾乎等於 source image 的重建，無法反映 target prompt
```

所有結構保留都透過**離散 token 替換**（P2P 邏輯）實現，不依賴連續特徵注入。

### 5. Single-Focus Fallback 原理（Phase 1.7 連續特徵注入）

add_object 等任務中，`source_focus_words=""` 但 `target_focus_words` 非空。此時：

```
single_focus_fallback = has_source_focus XOR has_target_focus = True
single_focus_side = "target"
```

**問題**：Phase 1.7 若純自由生成（`inject_image_features=None`），產出圖像與 source image 結構無關聯。target attention map 反映的是一張「隨機組合」的空間佈局，無法正確定位 target focus words 在 source 上的編輯位置。

**解法**：Phase 1.7 fallback 改用**連續特徵注入**（非離散 token 替換）：

```python
inject_schedule = [0.0] * phase17_fallback_replace_scales   # 前 M scale → 100% source image
                + [1.0] * (total_scales - M)                  # 後面 scale → target 自由生成

gen_one_img(
    inject_image_features=image_raw_features,   # 連續 VAE encoder 輸出
    inject_schedule=inject_schedule,
    p2p_token_storage=None,                      # 不做離散 token 替換
)
```

**為什麼用連續特徵而不是離散 token**：Phase 1 source gen 的結構保留機制是 `summed_codes` 層級的連續特徵注入。離散 token 只影響 `idx_Bld`（KV cache 同步用），但真正驅動下一個 scale input 的是 `summed_codes`。若只替換 token 而不注入特徵，`summed_codes` 仍是純自由生成的結果，視覺上看不到結構參考。

### 6. Focus Mask 計算流程

```python
# 對每個 scale >= num_full_replace_scales：
for block_idx in attn_block_indices:
    attn_map = extractor.extract_word_attention(
        block_idx=block_idx,
        scale_idx=si,
        token_indices=focus_token_indices,  # focus_words 的 T5 token indices
        spatial_size=(h, w),
    )
    # attn_map shape: (H, W)

# IQR 過濾離群 block（增加穩健性）
filtered_attn = iqr_filtered_mean(all_block_attn_maps)

# 百分位數二值化
threshold = np.percentile(filtered_attn, attn_threshold_percentile)
focus_mask = filtered_attn >= threshold  # True = focus 區域（不替換）
replacement_mask = ~focus_mask           # True = 背景（替換 source token）
```

### 7. 雙路 Mask 合併（P2P-Edit 特有）

```python
# combine_and_store_masks():
union_focus = source_focus_mask | target_focus_mask  # focus 區 = 兩者的聯集（OR）
replacement_mask = ~union_focus                       # 背景 = focus 之外的區域

# 語義：
# union 中任一 prompt 認為是「focus 的地方」都不被替換
# 確保 source focus 詞（如 "space suit"）和
# target focus 詞（如 "cupcake"）的位置都能自由生成
```

### 8. 各 scale 的 Token 替換邏輯

```python
# infinity_p2p_edit.py: autoregressive_infer_cfg 內

if si < p2p_attn_full_replace_scales:
    # 前 N scale：100% 全域替換（使用 source image token）
    idx_Bld = source_indices          # storage.tokens[si] = image 直接量化

elif p2p_use_mask and storage.has_mask_for_scale(si):
    # scale >= N：attention 遮罩替換
    # mask=True（背景）→ 替換為 source token（保留）
    # mask=False（focus）→ 保留 target 自由生成（改變）
    idx_Bld = torch.where(spatial_mask, source_indices, idx_Bld)

# 注意：無 focus words 時 p2p_token_replace_prob=0.0，Branch C 不啟用
# target 對 scale >= N 的部分完全自由生成（不做任何替換）
```

### 9. CFG 與 Attention Batch Index

在 Classifier-Free Guidance（CFG）下，模型同時生成兩個 batch：
- `batch[0]` — **conditioned**（prompt 驅動）← 用於 focus token 分析（`attn_batch_idx=0`）
- `batch[1]` — **unconditioned**（null prompt 驅動）

只有 conditioned batch 的 attention 才能反映 focus_words 對應的空間位置。

---

## 操作模式

### 模式一：P2P-Edit（有 source_image）

```bash
source_image="./imgs/bird.jpg"
source_focus_words="flowers"
target_focus_words="leaves"
```

流程：
1. Phase 0：encode source image（兩種路徑）
2. Phase 1：source gen + image features 注入 + attention 擷取
3. Phase 1.5：雙向 mask 計算（高 attention focus + 低 attention preserve）
4. Phase 1.6：建立 phase17_storage（low-attn 區域錨定）
5. Phase 1.7：引導 target 生成（preserve 區接地，focus 區自由）→ target attention
6. Phase 1.9：合併 mask + token 覆寫
7. Phase 2：target gen（P2P token 來自 image 量化 + combined attention 遮罩）

### 模式二：純 P2P-Attn（無 source_image）

```bash
source_image=""   # 留空
```

流程：
1. Phase 0：跳過
2. Phase 1：source gen（純文字驅動）+ attention 擷取
3. Phase 1.5：focus mask 計算（preserve mask 因無 image_scale_tokens 而跳過）
4. Phase 1.6：phase17_storage = None（跳過）
5. Phase 1.7：純 free-gen（無 preserve 錨定）→ target attention
6. Phase 1.9：合併 mask（token 不覆寫，使用 source gen 採樣）
7. Phase 2：target gen（P2P token 來自 source gen + attention 遮罩）

### 模式三：Single-Focus Fallback（add_object 等場景）

```bash
source_image="./imgs/room.jpg"
source_focus_words=""              # 無 source focus（add_object：source 沒有要移除的物件）
target_focus_words="with a crown"  # 只有 target focus
phase17_fallback_replace_scales=4
```

流程：
1. Phase 0：encode source image（兩種路徑）
2. Phase 1：source gen + image features 注入（attention 擷取跳過，因無 source focus）
3. Phase 1.5：跳過（無 source focus → 無 source_text_masks / source_low_attn_masks）
4. Phase 1.6：跳過（phase17_storage = None）
5. **Phase 1.7 Fallback**：target prompt + source image **連續特徵注入前 4 scale**（`inject_schedule=[0.0]*4 + [1.0]*9`），僅擷取 target attention
6. Phase 1.9：僅使用 target_text_masks + token 覆寫
7. Phase 2：target gen（P2P token 來自 image 量化 + target attention 遮罩）

### 模式四：無 focus words（target 完全自由）

```bash
source_focus_words=""
target_focus_words=""
auto_focus_from_prompt_diff=0
```

行為：
- `masks_stored == 0`
- Phase 2：`p2p_token_replace_prob=0.0`
- scale 0 ~ N-1：100% source token 替換
- scale N ~ end：target **完全自由生成**，不做任何替換

---

## 參數調整速查

### 結構保留太差？

```bash
# 增加全域替換的 scale 數
num_full_replace_scales=4

# 延長 source image 注入範圍
inject_weights="0.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0"
```

### Focus 區域未被改變？

```bash
# 降低 attention 閾值（擴大 focus 區域，更多位置允許自由生成）
attn_threshold_percentile=50

# 減少全域替換 scale（讓 attention 遮罩更早生效）
num_full_replace_scales=1
```

### Focus 區域太大（背景也被改變）？

```bash
# 提高 attention 閾值（縮小 focus 區域，更精確定位）
attn_threshold_percentile=85
```

### 確認遮罩是否正確？

```bash
# 查看遮罩視覺化
ls outputs/p2p_edit_*/attn_masks/

# combined/ 中的黑色區域 = focus union = target 自由生成的位置
# 如果黑色區域 ≠ 預期的 focus 詞位置，調整 attn_threshold_percentile
```

### 想切換為純 P2P-Attn 模式？

```bash
# 在 infer_p2p_edit.sh 中：
source_image=""   # 留空即可，其餘參數不變
```

---

## 輸出檔案說明

```
outputs/p2p_edit_YYYYMMDD_HHMMSS/
├── source.jpg                       Source 重建圖像（前 N scale 含 image injection）
├── target.jpg                       編輯後圖像（P2P-Attn 遮罩引導）
├── phase17_guided.jpg               Phase 1.7 引導生成圖像（debug_mode=1 時，一般模式）
├── phase17_fallback.jpg             Phase 1.7 fallback 生成圖像（debug_mode=1 時，single-focus）
├── source_case_dir → ...            符號連結，指回原始 PIE-Bench 案例資料夾（批量模式）
├── attn_masks/
│   ├── source/                      Source 高 attention focus 遮罩
│   │   ├── source_focus_scale04_8x8.png
│   │   ├── source_focus_scale05_16x16.png
│   │   ├── ...                      各 scale 個別遮罩
│   │   └── overlay.png              疊加圖（黑色=focus 不替換）
│   ├── phase17_preserve/            Phase 1.7 低 attention preserve 遮罩 ★
│   │   ├── preserve_scale04_8x8.png
│   │   ├── ...
│   │   └── overlay.png              白色=錨定區域（source image token 強制保留）
│   ├── target/                      Target focus 遮罩（Phase 1.7 引導生成的 attention）
│   │   ├── target_focus_scale04_8x8.png
│   │   ├── ...
│   │   └── overlay.png              疊加圖（黑色=focus 不替換）
│   └── combined/                    最終使用的合併 replacement 遮罩
│       ├── combined_focus_scale04_8x8.png
│       ├── ...
│       └── overlay.png              疊加圖（亮=背景替換，暗=focus union 保護區）
└── attention_cache/                 Attention cache 視覺化（use_attn_cache=1 時）
    └── alignments.json              Source→Target token 對齊資訊
```

**遮罩視覺化規則**：
- `source_focus_*`（`invert=True`）：黑色 = source focus 區（改變），白色 = 背景（替換）
- `phase17_preserve/`（`invert=False`）：白色 = low-attn 錨定區域（source token），黑色 = focus 自由區
- `target_focus_*`（`invert=True`）：黑色 = target focus 區（改變），白色 = 背景（替換）
- `combined_focus_*`（`invert=True`）：白色 = 背景替換區（source token），黑色 = focus union 保護區（target 自由）

啟用 `--use_cumulative_prob_mask 1` 時，`attn_masks/combined/` 會輸出 `combined_replace_prob_*`，
灰階值代表每個位置使用 source token 的替換機率（0~1）。

---

## 與其他管線的比較

| 特性 | `run_p2p.py` | `run_p2p_attn.py` | `run_p2p_edit.py` | `run_pie_edit.py` |
|------|-------------|-----------------|-----------------|-----------------|
| 適用場景 | 物件替換（整體結構保留）| 局部語義替換（文字/詞彙）| 有 source image 的局部語義替換 | PIE-Bench 評估（單一/批量）|
| Source image 輸入 | 不需要 | 不需要 | 可選（預設 None）| 必要 |
| Token 替換來源 | Source gen 採樣 | Source gen 採樣 | Source image 直接量化 ★ | Source image 直接量化 ★ |
| Focus mask | 無 | 單路（source only）| 三路（source focus + source preserve + target union）★ | 三路 + PIE mask + attn cache |
| Source gen 引導 | 純文字 | 純文字 | Image features 注入 summed_codes ★ | 同 P2P-Edit |
| Phase 1.7 single-focus | N/A | N/A | Source image 連續特徵注入 ★ | 同 P2P-Edit |
| Phase 2 image injection | 無 | 無 | 無（固定 None）| 無（固定 None）|
| PIE mask 支援 | 無 | 無 | 無 | 三種模式（0/1/2）★ |
| Attention cache | 無 | 無 | 無 | blended / full_p2p ★ |
| 無 focus words 行為 | 機率替換 | 機率替換 | 完全自由生成 ★ | 完全自由生成 ★ |
| Debug mode | 無 | 無 | `phase17_target.jpg` | `phase17_guided.jpg` / `phase17_fallback.jpg` |

★ = P2P-Edit 獨有特性

---

## 常見問題

### Q：沒有 source image 也可以使用嗎？

可以。將 `source_image=""` 留空，管線自動切換為純 P2P-Attn 模式，
行為與 `run_p2p_attn.py` 完全相同，但新增了雙路 mask（source + target focus 合併）。

### Q：focus_words 找不到對應 token

T5 SentencePiece 分詞可能將詞彙拆分為多個 sub-token，或與 prompt 片段不完全對齊。

**解法**：
1. 優先用逗號分隔 phrase（如 `"space suit, helmet"`），避免一次輸入過長片段
2. 嘗試只填關鍵詞（如 `"space suit"` 而非 `"wearing space suit"`）
3. 需完全手動控制時，加入 `--auto_focus_from_prompt_diff 0`，避免自動差異詞彙影響 focus set
4. 若仍無法找到，該側（source 或 target）的 mask 會自動跳過

### Q：target 圖像幾乎等於 source image

可能原因：`image_injection_scales` 或 `num_full_replace_scales` 太大，
強制太多 scale 都從 source image 複製，target prompt 的影響不足。

**解法**：
- 降低 `image_injection_scales`（如 1 或 2）
- 降低 `num_full_replace_scales`（如 1 或 2）
- 降低 `attn_threshold_percentile`（擴大允許 target 自由生成的區域）

### Q：target focus 區域仍殘留 source 結構

**解法**：
- 降低 `attn_threshold_percentile`（擴大 focus 保護區）
- 確認 `target_focus_words` 的詞彙有被正確 tokenize
- 調小 `attn_block_start`（使用更多 block 的 attention，增加 focus 覆蓋率）

### Q：背景與 source image 有明顯差異

**解法**：
- 提高 `attn_threshold_percentile`（縮小 focus 區域，減少背景誤識別）
- 增加 `num_full_replace_scales`（讓更多 scale 做 100% 結構替換）
- 增加 `image_injection_scales`（讓更多 scale 的 source gen 貼近 source image）

---

## 相關檔案

- **P2P 基礎管線**：[P2P_README.md](P2P_README.md)
- **P2P-Attn 管線**：[P2P_ATTN_README.md](P2P_ATTN_README.md)
- **快速上手**：[P2P_EDIT_QUICKSTART.md](P2P_EDIT_QUICKSTART.md)
- **BitwiseTokenStorage**：`infinity/utils/bitwise_token_storage.py`
- **CrossAttentionExtractor**：`attention_map/extractor.py`
- **P2P-Edit 基礎引擎**：`tools/run_p2p_edit.py`
- **PIE-Bench 評估器**：`tools/run_pie_edit.py`（單一 + 批量模式）
- **PIE-Bench 批量執行器**：`tools/batch_run_pie_edit.py`
- **P2P-Edit 模型**：`infinity/models/infinity_p2p_edit.py`
- **腳本**：`scripts/infer_p2p_edit.sh`、`scripts/single_infer_pie_edit.sh`、`scripts/batch_run_pie_edit.sh`
