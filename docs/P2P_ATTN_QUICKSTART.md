# P2P-Attn 快速上手指南

## 30 秒開始

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/infer_p2p_attn.sh
```

這將會：
1. 生成 source 圖像（告示牌寫著 "PLEASE STAND BEHIND LINE"）
2. 擷取 cross-attention map，定位文字區域
3. 計算 attention-based 空間遮罩
4. 生成 target 圖像（告示牌寫著 "DESTINATION: LONDON"，背景相同）

結果存於 `./outputs/p2p_attn/`

---

## 新建的檔案

```
InfLoop/
├── infinity/models/
│   └── infinity_p2p_attn.py        ← 新增：P2P-Attn 版模型
├── tools/
│   └── run_p2p_attn.py             ← 新增：P2P-Attn 主程式
├── scripts/
│   └── infer_p2p_attn.sh           ← 新增：快速執行腳本
└── docs/
    ├── P2P_ATTN_README.md          ← 新增：完整說明文件
    └── P2P_ATTN_QUICKSTART.md      ← 本文件
```

---

## 與原始 P2P 的差異

```
原始 P2P：
  Scale 0~N   → 100% 替換（全域結構保留）
  Scale N+1~  → 不替換（target 自由生成）
  問題：文字也被固定在 source 的版本

P2P-Attn：
  Scale 0~N   → 100% 替換（全域結構保留）    ← 相同
  Scale N+1~  → Attention 遮罩：
      文字區域（高 attention）→ 不替換         ← 新增
      背景區域（低 attention）→ 替換 source    ← 新增
```

---

## 各檔案職責

### `infinity/models/infinity_p2p_attn.py`

從 `infinity_p2p.py` 修改，主要差異：

1. **更新 docstring**（中文說明）
2. **新增參數** `p2p_attn_full_replace_scales`
3. **修改 token 儲存邏輯**：儲存全部 scale（非只前 N 個）
4. **修改 token 替換邏輯**：三分支
   - 分支 A：`si < p2p_attn_full_replace_scales` → 100% 替換
   - 分支 B：`si >= N` 且有 attention 遮罩 → 空間選擇性替換
   - 分支 C：Fallback → 機率替換

### `tools/run_p2p_attn.py`

主要新增功能（相較於 `run_p2p.py`）：

| 函式 | 功能 |
|------|------|
| `find_focus_token_indices()` | 在 source prompt 中找 focus_words 的 T5 token indices |
| `compute_attention_mask_for_scale()` | 計算單一 scale 的 attention-based 空間遮罩 |
| `_iqr_filtered_mean()` | IQR 過濾離群 block 後求平均（增加穩健性）|
| `build_and_store_attention_masks()` | 批次計算並存入所有 scale 的遮罩 |

主程式新增流程：
1. **Phase 0**：尋找 focus token indices（分詞匹配）
2. **Phase 1**：Source 生成 + CrossAttentionExtractor 掛載
3. **Phase 1.5**：計算 + 儲存 attention 遮罩（含視覺化）
4. **Phase 2**：Target 生成（attention 遮罩引導）

### `scripts/infer_p2p_attn.sh`

關鍵參數：

```bash
focus_words="PLEASE STAND BEHIND LINE"  # ← 修改這裡
source_prompt="..."
target_prompt="..."
num_full_replace_scales=4               # 前 N scale 全局替換
attn_threshold_percentile=75            # 閾值（前 25% 為文字區域）
```

---

## 自訂 Prompt 的修改步驟

**範例：貓咪咖啡店招牌，把 "LATTE" 改為 "ESPRESSO"**

```bash
# 1. 修改 scripts/infer_p2p_attn.sh

source_prompt="A cozy coffee shop chalkboard that reads \"LATTE\" in handwritten letters."
target_prompt="A cozy coffee shop chalkboard that reads \"ESPRESSO\" in handwritten letters."
focus_words="LATTE"                   # 要替換的詞
num_full_replace_scales=4             # 可調整
attn_threshold_percentile=75          # 可調整
save_file="./outputs/p2p_attn_cafe/"  # 修改輸出目錄

# 2. 執行
bash scripts/infer_p2p_attn.sh
```

---

## 參數調整速查

### 背景結構保留太差？

```bash
# 增加 num_full_replace_scales（更多 scale 做 100% 替換）
num_full_replace_scales=6

# 或提高 attn_threshold_percentile（縮小文字區域判定，增加替換範圍）
attn_threshold_percentile=85
```

### 文字仍未正確改變？

```bash
# 減少 num_full_replace_scales（讓更多 scale 可以自由生成）
num_full_replace_scales=3

# 或降低 attn_threshold_percentile（擴大文字區域，確保不被替換）
attn_threshold_percentile=60
```

### 想查看 attention 遮罩是否正確？

```bash
# 查看輸出的遮罩視覺化（白色 = 替換區域，黑色 = 文字區域）
ls outputs/p2p_attn/attn_masks/

# 如果 scale_12.png 中告示牌區域是黑色 → 正確
# 如果告示牌是白色 → 需要降低 attn_threshold_percentile
```

---

## 輸出檔案說明

```
outputs/p2p_attn/
├── source.jpg                   原始圖像（source prompt 生成）
├── target.jpg                   編輯後圖像（文字已替換）
├── tokens_p2p_attn.pkl          Token + 遮罩資料（可重複使用）
└── attn_masks/
    ├── replacement_mask_scale04_8x8.png     Scale 4 遮罩（白=替換）
    ├── replacement_mask_scale05_16x16.png   Scale 5 遮罩
    ├── replacement_mask_scale06_24x24.png   Scale 6 遮罩
    └── ...                                   更多 scale 的遮罩
```

---

## 與其他腳本的相容性

| 腳本 | 模型檔案 | 適用場景 |
|------|---------|---------|
| `infer.sh` | `infinity.py` | 一般圖像生成 |
| `infer_p2p.sh` | `infinity_p2p.py` | 物件替換（整體結構一致）|
| `infer_p2p_attn.sh` | `infinity_p2p_attn.py` | **局部文字替換**（本管線）|
