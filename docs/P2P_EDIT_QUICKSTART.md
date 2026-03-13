# P2P-Edit 快速上手指南

## 30 秒開始

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/infer_p2p_edit.sh
```

這將會：
1. 載入 source image 並進行兩種 VAE 編碼（連續特徵 + 離散 token）
2. 生成 source 圖像（前幾個 scale 注入 source image 的 VAE features）
3. 從同一組 source attention 計算兩種遮罩：**高 attention focus mask** + **低 attention preserve mask**（不需額外 forward pass）
4. 建立 Phase 1.7 preserve storage：低 attention 背景區 → 強制使用 source image token 錨定
5. 引導 target 生成（Phase 1.7）：preserve 區域接地 source image，focus 區域 target 自由生成 → 收集 target attention
6. 合併 source focus + target focus → replacement mask，以 source image token 覆寫 storage
7. 生成 target 圖像（P2P token 替換來自 source image 直接量化，combined attention 遮罩引導）

結果存於 `./outputs/p2p_edit_YYYYMMDD_HHMMSS/`

---

## 新建的檔案

```
InfLoop/
├── infinity/models/
│   └── infinity_p2p_edit.py        ← 新增：P2P-Edit 版模型（支援連續特徵注入）
├── tools/
│   └── run_p2p_edit.py             ← 新增：P2P-Edit 主程式
├── scripts/
│   └── infer_p2p_edit.sh           ← 新增：快速執行腳本
└── docs/
    ├── P2P_EDIT_README.md          ← 新增：完整說明文件
    └── P2P_EDIT_QUICKSTART.md      ← 本文件
```

---

## 與 P2P-Attn 的差異

```
P2P-Attn（run_p2p_attn.py）：
  Phase 0   → 無（不需要 source image）
  Token 來源 → Source gen 採樣（受 source prompt 語義影響）
  Focus mask → 單路（source prompt → source attention）
  Phase 1.7  → 純 free-gen（無錨定）

P2P-Edit（run_p2p_edit.py）：
  Phase 0   → Source image → VAE encoder →
                (A) 連續特徵（注入 source gen summed_codes）
                (B) 離散 token（覆寫 storage，供 Phase 2 P2P 替換）
  Token 來源 → Source image 直接量化（精確反映像素）    ← 關鍵差異
  Focus mask → 三路（source focus + source preserve + target union）★
  Phase 1.5  → 高 attention focus mask + 低 attention preserve mask（同一次計算）
  Phase 1.6  → phase17_storage：preserve 區域錨定 source image token
  Phase 1.7  → 引導生成：preserve 區接地，focus 區 target 自由 ← 關鍵差異
  Phase 2   → 永不注入 source image（確保 target 可自由改變內容）
```

---

## 各檔案職責

### `infinity/models/infinity_p2p_edit.py`

從 `infinity_p2p_attn.py` 修改，主要差異：

1. **新增 `inject_image_features` 參數**：接收連續 VAE encoder 輸出
2. **新增 `inject_schedule` 參數**：各 scale 的注入強度（0.0=100% image，1.0=自由）
3. **summed_codes 注入邏輯**：
   ```python
   summed_codes = lerp(image_features_for_scale, gen_codes, inject_schedule[si])
   ```
4. **Token 替換邏輯**（三分支，與 P2P-Attn 相同）：
   - 分支 A：`si < p2p_attn_full_replace_scales` → 100% 替換（使用 image 量化 token）
   - 分支 B：`p2p_use_mask` 且 has_mask → attention 遮罩替換
   - 分支 C：`p2p_token_replace_prob > 0` → 機率替換（無遮罩 fallback）

### `tools/run_p2p_edit.py`

六個執行階段的主程式。新增（相較於 `run_p2p_attn.py`）：

| 函式 / 區塊 | 功能 |
|-------------|------|
| `encode_image_to_raw_features()` | Source image → 連續 VAE encoder 輸出（用於 summed_codes 注入）|
| `encode_image_to_scale_tokens()` | Source image → 離散 bit token（用於覆寫 storage.tokens）|
| Phase 0 | 載入 source image，進行兩種編碼，建立 inject_schedule |
| Phase 1.5 | 從 source attention 同時計算 focus mask（高 attention）+ preserve mask（低 attention）|
| Phase 1.6 | 建立 phase17_storage：low-attn 區域 → source image token 錨定 |
| Phase 1.7 | **引導** target 生成：preserve 區域接地 source image，focus 區域 target 自由 |
| Phase 1.9 | 雙路 mask 合併 + **覆寫 storage.tokens**（P2P-Edit 核心步驟）|
| `--source_image` CLI 參數 | 可選（default=None），留空退回 P2P-Attn 模式 |

### `scripts/infer_p2p_edit.sh`

關鍵參數：

```bash
source_image="./image1.png"        # ← 設定 source image 路徑（留空 = 純 P2P-Attn 模式）
source_prompt="..."
target_prompt="..."
source_focus_words="with a Pearl Earring"  # ← source prompt 中欲替換的詞彙
target_focus_words=""              # ← target prompt 中對應的新詞彙（可空）
image_injection_scales=2           # ← 前幾個 scale 注入 source image（連續特徵層級）
num_full_replace_scales=2          # ← 前幾個 scale 100% 替換（離散 token 層級）
attn_threshold_percentile=80       # ← attention 閾值（80 = 最高 20% 為 focus；最低 20% 為 preserve）
attn_block_start=2                 # ← 從第 3 個 block 開始計算 attention
```

---

## 自訂 Prompt 的修改步驟

**範例：移除狗身上的太空裝**

```bash
# 1. 修改 scripts/infer_p2p_edit.sh

source_image="./imgs/dog_spacesuit.jpg"
source_prompt="a dog wearing space suit"
target_prompt="a dog"
source_focus_words="space suit"
target_focus_words=""
image_injection_scales=2
num_full_replace_scales=2
attn_threshold_percentile=50
save_file="./outputs/p2p_edit_dog/"  # 可手動指定輸出目錄

# 2. 執行
bash scripts/infer_p2p_edit.sh
```

**範例：替換物件（鳥→杯子蛋糕）**

```bash
source_image="./imgs/bird.jpg"
source_prompt="a wathet bird sitting on a branch of yellow flowers"
target_prompt="a cupcake sitting on a branch of yellow flowers"
source_focus_words="wathet bird"
target_focus_words="cupcake"
attn_threshold_percentile=50
```

**範例：純 P2P-Attn 模式（無 source image）**

```bash
source_image=""   # 留空即可，其餘不變
source_prompt="A cozy coffee shop chalkboard that reads \"LATTE\" in handwritten letters."
target_prompt="A cozy coffee shop chalkboard that reads \"ESPRESSO\" in handwritten letters."
source_focus_words="LATTE"
target_focus_words="ESPRESSO"
```

---

## 參數調整速查

### 結構保留太差？

```bash
# 增加全域替換的 scale 數
num_full_replace_scales=4

# 延長 source image 注入範圍，或手動指定 inject_weights
inject_weights="0.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0"
```

### Focus 區域未改變？

```bash
# 降低 attention 閾值（擴大 focus 區域）
attn_threshold_percentile=40

# 減少全域替換 scale
num_full_replace_scales=1
```

### 確認遮罩是否對準物件？

```bash
# 查看遮罩（統一規則：白色=保留，黑色=改變）
ls outputs/p2p_edit_*/attn_masks/

# combined/replacement_overlay.png 中：
#   黑色區域 = focus union = target 自由生成的位置
# 如果黑色 ≠ 預期的物件位置 → 調整 attn_threshold_percentile
```

---

## 輸出檔案說明

```
outputs/p2p_edit_YYYYMMDD_HHMMSS/
├── source.jpg                     Source 重建圖像（前 N scale 含 image injection）
├── target.jpg                     編輯後圖像（P2P-Attn focus 遮罩引導）
├── attn_masks/
│   ├── source/                    Source focus 遮罩（黑色=focus 區不替換）
│   │   ├── source_focus_scale04_*.png
│   │   └── overlay.png                ← 疊加圖，黑色=focus 不替換
│   ├── phase17_preserve/          Phase 1.7 低 attention preserve 遮罩 ★ 新增
│   │   └── overlay.png                ← 白色=錨定區域（source token 強制保留）
│   ├── target/                    Target focus 遮罩（Phase 1.7 引導生成的 attention）
│   │   └── overlay.png
│   └── combined/                  合併 replacement 遮罩（最終 Phase 2 使用）
│       └── overlay.png                ← 最重要：白色=source 替換，黑色=union 保護區
└── tokens_p2p_edit.pkl             Token + 遮罩資料
```

**遮罩視覺化規則**：

| 遮罩 | 白色 | 黑色 |
|------|------|------|
| `source_focus_*` | 背景替換區 | Source focus 區（target 自由生成）|
| `phase17_preserve/` | **低 attention 錨定區**（source image token 強制保留）| Focus 自由區 |
| `target_focus_*` | 背景替換區 | Target focus 區（改變）|
| `combined_focus_*` | 背景（source token 替換）| Union focus 區（target 自由）|

---

## 與其他腳本的相容性

| 腳本 | 模型檔案 | 適用場景 |
|------|---------|---------|
| `infer.sh` | `infinity.py` | 一般圖像生成 |
| `infer_p2p.sh` | `infinity_p2p.py` | 物件替換（seed 相同結構）|
| `infer_p2p_attn.sh` | `infinity_p2p_attn.py` | 局部語義替換（無 source image）|
| `infer_p2p_edit.sh` | `infinity_p2p_edit.py` | **有 source image 的局部語義替換**（本管線）|
