# P2P-Attn 管線（Attention-Guided Prompt-to-Prompt）

## 概述

P2P-Attn 是在 P2P 管線基礎上的進化版本，透過 **cross-attention 空間遮罩** 實現更精確的局部文字替換，同時保留整體場景結構。

### 核心設計理念

| 場景 | 問題 | P2P-Attn 解法 |
|------|------|--------------|
| 原始 P2P | 前 N scale 全部替換 → 文字區域被固定，target 無法渲染新文字 | 在深層 scale 加入 attention 遮罩，讓文字位置自由生成 |

### 運作示意

```
Ps: A train platform sign that reads "PLEASE STAND BEHIND LINE" as a train approaches.
Pt: A train platform sign that reads "DESTINATION: LONDON" as a train approaches.
                                       ↑ focus_words ↑
```

**結果**：
- 月台、人群、列車背景 → 與 source image 完全相同（結構保留）
- 告示牌文字區域 → 由 target prompt 自由生成「DESTINATION: LONDON」

---

## 架構

```
InfLoop/
├── infinity/
│   ├── models/
│   │   ├── infinity_p2p_attn.py      # P2P-Attn 版模型（本次新增）
│   │   └── infinity_p2p.py           # 原始 P2P 版模型（基礎）
│   └── utils/
│       └── bitwise_token_storage.py  # Token + 遮罩儲存（共用）
├── tools/
│   └── run_p2p_attn.py               # P2P-Attn 主程式（本次新增）
├── scripts/
│   └── infer_p2p_attn.sh             # 快速執行腳本（本次新增）
├── attention_map/
│   └── extractor.py                  # CrossAttentionExtractor（現有，共用）
└── docs/
    ├── P2P_ATTN_README.md            # 本文件
    └── P2P_ATTN_QUICKSTART.md        # 快速上手
```

---

## 工作流程

### 三個階段

```
┌─────────────────────────────────────────────────────────┐
│           P2P-Attn 管線（三個階段）                      │
└─────────────────────────────────────────────────────────┘

┌── Phase 1：Source 生成 ─────────────────────┐
│                                            │
│  Source Prompt                             │
│      ↓                                     │
│  [CrossAttentionExtractor 掛載]            │
│  Infinity 模型自回歸生成（全部 scale）      │
│  ├─ 擷取每個 scale 的 cross-attention map  │
│  └─ 儲存所有 scale 的 bitwise token        │
│      ↓                                     │
│  Source Image + Token Storage              │
└────────────────────────────────────────────┘
         ↓
┌── Phase 1.5：遮罩計算 ──────────────────────┐
│                                            │
│  Focus Words = "PLEASE STAND BEHIND LINE"  │
│      ↓                                     │
│  在 source prompt 中找 T5 token indices    │
│      ↓                                     │
│  對每個 scale >= num_full_replace_scales：  │
│  ├─ 取各 block 的 attention map             │
│  ├─ IQR 過濾離群 block                     │
│  ├─ 對 focus token indices 求平均          │
│  ├─ 百分位數閾值二值化                     │
│  │   高 attention = 文字區域 (False)       │
│  │   低 attention = 背景區域 (True)        │
│  └─ 存入 BitwiseTokenStorage.masks         │
└────────────────────────────────────────────┘
         ↓
┌── Phase 2：Target 生成 ─────────────────────┐
│                                            │
│  Target Prompt                             │
│      ↓                                     │
│  Infinity 模型自回歸生成                   │
│                                            │
│  Scale 0 ~ N-1（全域替換）：               │
│      100% 替換為 source token              │
│      （保留整體佈局和粗略結構）            │
│                                            │
│  Scale N ~ end（Attention 引導）：         │
│      ├─ 背景區域（低 attention, mask=True）│
│      │   → 替換為 source token             │
│      │   （保留月台、人群、列車等）        │
│      └─ 文字區域（高 attention, mask=False)│
│          → 保留 target 自由生成            │
│          （渲染 "DESTINATION: LONDON"）    │
└────────────────────────────────────────────┘
```

---

## 快速開始

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/infer_p2p_attn.sh
```

結果儲存於 `outputs/p2p_attn/`：
- `source.jpg` — 原始圖像
- `target.jpg` — 文字替換後的圖像
- `attn_masks/` — Attention 遮罩視覺化（白=替換區域，黑=文字區域）
- `tokens_p2p_attn.pkl` — Token + 遮罩資料

---

## 參數說明

### 核心參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--source_prompt` | 必填 | Source prompt（含欲修改的文字）|
| `--target_prompt` | 必填 | Target prompt（含新文字）|
| `--focus_words` | 必填 | Source prompt 中欲關注的詞彙（空格分隔）<br>例：`"PLEASE STAND BEHIND LINE"` |
| `--num_full_replace_scales` | `4` | 前幾個 scale 做 100% source token 替換 |
| `--attn_threshold_percentile` | `75` | Attention 閾值（前 N% 為文字區域）|
| `--save_file` | `./outputs/p2p_attn` | 輸出目錄 |

### 進階調整參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--attn_block_start` | `-1` | Attention 計算的起始 block（-1=自動=後半段）|
| `--attn_block_end` | `-1` | Attention 計算的結束 block（-1=自動=最後）|
| `--attn_batch_idx` | `0` | CFG batch 索引（0=conditioned，對應 source prompt）|
| `--p2p_token_replace_prob` | `0.5` | Fallback 機率替換（無遮罩時使用）|
| `--save_attn_vis` | `1` | 是否儲存遮罩視覺化圖 |

---

## 技術細節

### 1. Token 儲存策略

**P2P（原版）**：僅儲存前 N 個 scale 的 token
```python
BitwiseTokenStorage(num_scales=num_source_scales)  # 只存前 N scale
```

**P2P-Attn（本版）**：儲存所有 scale 的 token
```python
BitwiseTokenStorage(num_scales=total_scales)  # 存全部 scale
```

原因：attention 遮罩需要對後續 scale 做精確的局部替換，需要 source 所有 scale 的 token。

### 2. Attention Map 的解讀

Cross-attention map 形狀：`[1, num_heads, query_len, key_len]`

- **query** = 空間位置（圖像 token）
- **key** = 文字 token

`attention[i, :, q, k]` = 空間位置 `q` 對文字 token `k` 的注意力強度

高值代表：這個空間位置與文字 token `k` 強相關 → 很可能是文字渲染區域。

### 3. 遮罩計算流程

```python
# 對每個 scale si >= num_full_replace_scales：
for block_idx in attn_block_indices:
    attn_map = extractor.extract_word_attention(
        block_idx=block_idx,
        scale_idx=si,
        token_indices=focus_token_indices,  # focus_words 的 T5 token indices
        spatial_size=(h, w),
    )
    # attn_map shape: (H, W) — 每個空間位置對 focus_words 的平均注意力

# IQR 過濾離群 block（增加穩健性）
filtered_attn = iqr_filtered_mean(all_block_attn_maps)

# 百分位數二值化
threshold = np.percentile(filtered_attn, attn_threshold_percentile)
text_mask = filtered_attn >= threshold  # True = 文字區域（不替換）
replacement_mask = ~text_mask           # True = 背景（替換 source token）

# 存入 storage
storage.masks[si] = replacement_mask  # [1, 1, h, w, 1], bool
```

### 4. Target 生成替換邏輯

```python
# infinity_p2p_attn.py: autoregressive_infer_cfg 內

if si < p2p_attn_full_replace_scales:
    # 前 N scale：100% 全域替換
    idx_Bld = source_indices  

elif p2p_use_mask and storage.has_mask_for_scale(si):
    # scale >= N：attention 遮罩替換
    # mask=True 的位置 → 替換為 source token（背景）
    # mask=False 的位置 → 保留 target 自由生成（文字區域）
    idx_Bld = torch.where(spatial_mask, source_indices, idx_Bld)

else:
    # Fallback：無遮罩時使用機率替換
    rand_mask = torch.rand(...) < p2p_token_replace_prob
    idx_Bld = torch.where(rand_mask, source_indices, idx_Bld)
```

### 5. CFG 與 Attention Batch Index

在 Classifier-Free Guidance（CFG）設定下，模型同時生成兩個 batch：
- `batch[0]` — **conditioned**（source prompt 驅動的生成）
- `batch[1]` — **unconditioned**（null prompt 驅動的生成）

對於 focus token 分析，應使用 **conditioned batch**（`attn_batch_idx=0`），
因為文字區域的 attention 只有在 conditioned prompt 下才具有意義。

---

## 調整建議

### `num_full_replace_scales` 的選擇

| 值 | 效果 |
|----|------|
| 2~3 | 只保留極粗略結構，給 target 較多創作空間 |
| 4~5 | **推薦**，平衡結構保留與文字自由度 |
| 6+ | 結構非常固定，但可能影響文字渲染 |

### `attn_threshold_percentile` 的選擇

| 值 | 文字區域 | 適用場景 |
|----|---------|---------|
| 50 | 較大（前 50%）| 文字很大、佔畫面比例高 |
| 75 | 適中（前 25%）| **推薦**，適合一般告示牌場景 |
| 90 | 較小（前 10%）| 文字很小、要精確定位 |

---

## 常見問題

### Q：focus_words 找不到對應 token

**原因**：T5 tokenizer 的 SentencePiece 分詞可能與 focus_words 不完全匹配。

**解法**：
1. 確認 focus_words 與 source_prompt 中的大小寫一致
2. 嘗試縮短 focus_words（只用最關鍵的詞）
3. 若仍無法找到，會自動 fallback 到機率替換

### Q：target 圖像中文字仍未改變

**可能原因**：
1. `num_full_replace_scales` 太大，文字 scale 被全部替換
2. `attn_threshold_percentile` 太高，文字區域被縮到極小
3. `attn_batch_idx` 設定有誤

**解法**：
- 降低 `num_full_replace_scales`（如從 5 降到 3）
- 降低 `attn_threshold_percentile`（如從 75 降到 60）
- 確認 `attn_batch_idx=0`

### Q：背景結構有明顯差異

**可能原因**：
- `attn_threshold_percentile` 太低，背景誤判為文字區域
- `num_full_replace_scales` 太小

**解法**：
- 提高 `attn_threshold_percentile`（如從 75 提高到 85）
- 提高 `num_full_replace_scales`

---

## 與 P2P 管線的比較

| 特性 | `run_p2p.py` | `run_p2p_attn.py` |
|------|-------------|-----------------|
| 適用場景 | 物件替換（狗→貓）| 局部文字替換（告示牌文字）|
| 後期 scale 策略 | 不替換 | Attention 遮罩選擇性替換 |
| Attention 擷取 | 否 | 是（CrossAttentionExtractor）|
| Token 儲存範圍 | 前 N scale | 全部 scale |
| 文字保真度 | 低（文字可能固定）| 高（文字區域自由生成）|
| 背景保真度 | 中（後期不替換）| 高（背景區域精確替換）|
