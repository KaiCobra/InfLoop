# P2P-Edit 管線（Source Image Injection + Attention-Guided P2P）

## 概述

P2P-Edit 是在 P2P-Attn 管線基礎上的進化版本，透過 **source image 直接 VAE 編碼注入** 與 **雙路 attention 空間遮罩**，實現在保持 source image 結構的同時，精確替換局部語義內容。

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
│   │   └── infinity_p2p_edit.py     # P2P-Edit 版模型（支援連續特徵注入 + P2P token 替換）
│   └── utils/
│       └── bitwise_token_storage.py # Token + 遮罩儲存（共用）
├── tools/
│   └── run_p2p_edit.py              # P2P-Edit 主程式（本管線）
├── scripts/
│   └── infer_p2p_edit.sh            # 快速執行腳本
├── attention_map/
│   └── extractor.py                 # CrossAttentionExtractor（現有，共用）
└── docs/
    ├── P2P_EDIT_README.md           # 本文件
    └── P2P_EDIT_QUICKSTART.md       # 快速上手
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
│  Target Prompt + phase17_storage（preserve mask）              │
│       ↓                                                         │
│  [Target CrossAttentionExtractor 掛載]                         │
│  Scale 0 ~ N-1：100% source image token（結構錨定）             │
│  Scale N+  ：                                                   │
│  ├─ 低 attention 背景區（preserve mask=True）                   │
│  │   → 強制使用 source image token（消除 source/model 衝突）    │
│  └─ 高 attention focus 區（preserve mask=False）                │
│      → target prompt 自由生成（反映 target 語義）               │
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

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/infer_p2p_edit.sh
```

結果儲存於 `./outputs/p2p_edit_YYYYMMDD_HHMMSS/`：
- `source.jpg` — Source 重建圖像（含 image injection）
- `target.jpg` — 編輯後圖像（P2P-Attn 引導）
- `attn_masks/source/` — Source focus 遮罩（黑色=focus 不替換，白色=背景替換）
- `attn_masks/phase17_preserve/` — Phase 1.7 低 attention preserve 遮罩（**白色=錨定區域**）★
- `attn_masks/target/` — Target focus 遮罩（黑色=focus 不替換，白色=背景替換）
- `attn_masks/combined/` — 合併 replacement 遮罩（**白色=source token 替換，黑色=union 保護區**）
- `tokens_p2p_edit.pkl` — Token + 遮罩資料

---

## 參數說明

### 核心 Prompt 參數

| 參數 | 說明 |
|------|------|
| `--source_prompt` | Source prompt（含欲修改的物件/詞彙）|
| `--target_prompt` | Target prompt（修改後的內容）|
| `--source_focus_words` | Source prompt 中欲關注的詞彙（空格分隔）|
| `--target_focus_words` | Target prompt 中對應的新詞彙（可為空字串）|

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
| `--attn_threshold_percentile` | `80` | 高於此百分位 = focus 區域（不替換）。值越小 = focus 區越大（80 = 最高 20% 為 focus；最低 20% 為 Phase 1.7 preserve）|
| `--p2p_token_replace_prob` | `0.0` | Fallback 機率替換（有遮罩才生效）；0.0 表示不啟用 |
| `--save_attn_vis` | `1` | 是否儲存遮罩視覺化 |

### Attention Block 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--attn_block_start` | `2` | 起始 block index（2 = 從第 3 個 block 開始；-1 = 自動使用後半段 block）|
| `--attn_block_end` | `-1` | 結束 block index（-1 = 自動，使用最後一個 block）|
| `--attn_batch_idx` | `0` | CFG 下擷取哪個 batch（0 = conditioned，對應 source/target prompt）|

> **`attn_threshold_percentile` 同時控制兩個閾值**：
> - 高 attention focus（Phase 1.5 ①、Phase 2 不替換）：最高 `(100 - p)%`
> - 低 attention preserve（Phase 1.5 ②、Phase 1.7 錨定）：最低 `(100 - p)%`
> 兩者對稱，各佔相同比例（如 p=80 → 各 20%）。

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

### 5. Focus Mask 計算流程

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

### 6. 雙路 Mask 合併（P2P-Edit 特有）

```python
# combine_and_store_masks():
union_focus = source_focus_mask | target_focus_mask  # focus 區 = 兩者的聯集（OR）
replacement_mask = ~union_focus                       # 背景 = focus 之外的區域

# 語義：
# union 中任一 prompt 認為是「focus 的地方」都不被替換
# 確保 source focus 詞（如 "space suit"）和
# target focus 詞（如 "cupcake"）的位置都能自由生成
```

### 7. 各 scale 的 Token 替換邏輯

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

### 8. CFG 與 Attention Batch Index

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

### 模式三：無 focus words（target 完全自由）

```bash
source_focus_words=""
target_focus_words=""
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
└── tokens_p2p_edit.pkl               Token + 遮罩資料（可重複使用）
```

**遮罩視覺化規則**：
- `source_focus_*`（`invert=True`）：黑色 = source focus 區（改變），白色 = 背景（替換）
- `phase17_preserve/`（`invert=False`）：白色 = low-attn 錨定區域（source token），黑色 = focus 自由區
- `target_focus_*`（`invert=True`）：黑色 = target focus 區（改變），白色 = 背景（替換）
- `combined_focus_*`（`invert=True`）：白色 = 背景替換區（source token），黑色 = focus union 保護區（target 自由）

---

## 與其他管線的比較

| 特性 | `run_p2p.py` | `run_p2p_attn.py` | `run_p2p_edit.py` |
|------|-------------|-----------------|-----------------|
| 適用場景 | 物件替換（整體結構保留）| 局部語義替換（文字/詞彙）| 有 source image 的局部語義替換 |
| Source image 輸入 | 不需要 | 不需要 | 可選（預設 None）|
| Token 替換來源 | Source gen 採樣 | Source gen 採樣 | Source image 直接量化 ★ |
| Focus mask | 無 | 單路（source only）| 三路（source focus + source preserve + target union）★ |
| Source gen 引導 | 純文字 | 純文字 | Image features 注入 summed_codes ★ |
| Phase 2 image injection | 無 | 無 | 無（固定 None）|
| 無 focus words 行為 | 機率替換 | 機率替換 | 完全自由生成 ★ |

★ = P2P-Edit 獨有特性

---

## 常見問題

### Q：沒有 source image 也可以使用嗎？

可以。將 `source_image=""` 留空，管線自動切換為純 P2P-Attn 模式，
行為與 `run_p2p_attn.py` 完全相同，但新增了雙路 mask（source + target focus 合併）。

### Q：focus_words 找不到對應 token

T5 SentencePiece 分詞可能將詞彙拆分為多個 sub-token，或大小寫不匹配。

**解法**：
1. 確認 focus_words 與 prompt 中的詞彙大小寫完全一致
2. 嘗試只填關鍵詞（如 `"space suit"` 而非 `"wearing space suit"`）
3. 若仍無法找到，`source_focus_words` 或 `target_focus_words` 各自為空時，會自動跳過對應的 mask 計算

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
- **P2P-Edit 模型**：`infinity/models/infinity_p2p_edit.py`
