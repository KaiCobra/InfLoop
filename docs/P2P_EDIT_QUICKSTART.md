# P2P-Edit 快速上手指南

## 30 秒開始

### 方式一：單一案例 P2P-Edit

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/infer_p2p_edit.sh
```

### 方式二：單一案例 PIE-Bench P2P-Edit（自動讀取 meta.json）

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

以方式一為例，這將會：
1. 載入 source image 並進行兩種 VAE 編碼（連續特徵 + 離散 token）
2. 生成 source 圖像（前幾個 scale 注入 source image 的 VAE features）
3. 從同一組 source attention 計算兩種遮罩：**高 attention focus mask** + **低 attention preserve mask**（不需額外 forward pass）
4. 建立 Phase 1.7 preserve storage：低 attention 背景區 → 強制使用 source image token 錨定
5. 引導 target 生成（Phase 1.7）：preserve 區域接地 source image，focus 區域 target 自由生成 → 收集 target attention
   - **Single-focus fallback**（如 add_object）：改用 source image **連續特徵注入**前 N scale（`--phase17_fallback_replace_scales=4`），target 在細節 scale 自由生成
6. 合併 source focus + target focus → replacement mask，以 source image token 覆寫 storage
7. 生成 target 圖像（P2P token 替換來自 source image 直接量化，combined attention 遮罩引導）

結果存於 `./outputs/p2p_edit_YYYYMMDD_HHMMSS/`

補充（目前程式預設）：
- `--auto_focus_from_prompt_diff=1`：會自動把 source/target prompt 差異片段加入 focus terms。
- `--use_cumulative_prob_mask=0`：預設使用單一 bool replacement mask；設為 1 可改成跨尺度機率替換。
- `--phase17_fallback_replace_scales=4`：single-focus 時 Phase 1.7 以 source image 連續特徵注入前 4 scale。
- `--debug_mode=0`：設為 1 可儲存所有中間階段圖片（`phase17_guided.jpg`、`phase17_fallback.jpg`）。

---

## 檔案架構

```
InfLoop/
├── infinity/models/
│   └── infinity_p2p_edit.py            P2P-Edit 版模型（支援連續特徵注入）
├── tools/
│   ├── run_p2p_edit.py                 P2P-Edit 基礎引擎（Phase 0~2 + gen_one_img）
│   ├── run_pie_edit.py                 PIE-Bench P2P-Edit 評估（單一 + 批量模式）
│   └── batch_run_pie_edit.py           PIE-Bench 批量執行器
├── scripts/
│   ├── infer_p2p_edit.sh               單一案例 P2P-Edit 執行腳本
│   ├── single_infer_pie_edit.sh        單一案例 PIE-Bench P2P-Edit（自動讀取 meta.json）
│   └── batch_run_pie_edit.sh           PIE-Bench 批量執行腳本
└── docs/
    ├── P2P_EDIT_README.md              完整說明文件
    └── P2P_EDIT_QUICKSTART.md          本文件
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
               ★ Single-focus fallback（如 add_object：source_focus=""）：
               → 改用 source image 連續特徵注入前 N scale（inject_image_features）
               → 後續 scale target 自由生成
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
| Prompt focus 準備 | 支援 `--auto_focus_from_prompt_diff`：自動從 source/target prompt 差異補 focus terms |
| Phase 1.5 | 從 source attention 同時計算 focus mask（高 attention）+ preserve mask（低 attention）|
| Phase 1.6 | 建立 phase17_storage：low-attn 區域 → source image token 錨定 |
| Phase 1.7 | **引導** target 生成：preserve 區域接地 source image，focus 區域 target 自由 |
| Phase 1.9 | 雙路 mask 合併 + **覆寫 storage.tokens**（P2P-Edit 核心步驟）|
| Mask 後處理 | 支援 `--use_cumulative_prob_mask`：可轉為跨尺度累積機率遮罩 |
| `--source_image` CLI 參數 | 可選（default=None），留空退回 P2P-Attn 模式 |
| `--debug_mode` | 設為 1 儲存所有中間階段圖片 |

### `tools/run_pie_edit.py`

PIE-Bench 評估整合器，支援**單一案例**和**批量模式**：

| 功能 | 說明 |
|------|------|
| 批量模式 | `--bench_dir` + `--output_dir`：遍歷全部 PIE-Bench 案例 |
| 單一案例模式 | `--source_image` + `--source_prompt` + `--target_prompt`（無需 bench_dir）|
| 自動讀取 | 讀取 `meta.json` 中的 source_prompt、target_prompt、focus_words |
| Phase 1.7 fallback | `single_focus_fallback` 判斷：改用 `inject_image_features` 連續特徵注入 |
| PIE mask | 支援 `--pie_use_mask=1` 載入外部遮罩（`mask85.txt` 比例為 85%）|
| Attention cache | `--pie_attn_cache_both=1` 快取 source/target attention，跳過重複計算 |
| `--phase17_fallback_replace_scales` | 控制 single-focus fallback 的結構注入 scale 數 |

### `tools/batch_run_pie_edit.py`

PIE-Bench **批量執行器**：調用 `run_pie_edit.py`，支援 schedule sweep 和多參數組合。

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
auto_focus_from_prompt_diff=1      # ← 自動補 source/target prompt 差異詞彙
use_cumulative_prob_mask=0         # ← 0=bool遮罩, 1=跨尺度累積機率遮罩
phase17_fallback_replace_scales=4  # ← single-focus fallback 時注入 source image 的 scale 數
debug_mode=1                       # ← 設為 1 儲存中間階段圖片
```

### `scripts/single_infer_pie_edit.sh`

PIE-Bench 單一案例暨快速測試腳本（自動讀取 `meta.json`）：

```bash
dir="./data/infinity_toy_data/000000000001"  # ← PIE-Bench 案例資料夾
# 自動讀取 meta.json 中的 source_prompt、target_prompt、focus_words
# 呼叫 tools/run_pie_edit.py（單一案例模式）
```

### `scripts/batch_run_pie_edit.sh`

PIE-Bench 批量執行 + schedule sweep：

```bash
bench_dir="./data/infinity_toy_data"
output_dir="./outputs/eval_pie"
schedules_json="./scripts/schedules_sweep.json"
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
auto_focus_from_prompt_diff=0      # 手動指定 focus 時建議關閉
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

### 想要更柔和的替換（機率式）？

```bash
# 啟用跨尺度累積機率遮罩
use_cumulative_prob_mask=1

# combined/ 會輸出 combined_replace_prob_*.png（灰階=替換機率）
```

---

## 輸出檔案說明

```
outputs/p2p_edit_YYYYMMDD_HHMMSS/
├── source.jpg                     Source 重建圖像（前 N scale 含 image injection）
├── target.jpg                     編輯後圖像（P2P-Attn focus 遮罩引導）
├── phase17_guided.jpg             ★ Phase 1.7 引導生成（debug_mode=1 時輸出）
├── phase17_fallback.jpg           ★ Phase 1.7 single-focus fallback 結果（debug_mode=1）
├── attn_masks/
│   ├── source/                    Source focus 遮罩（黑色=focus 區不替換）
│   │   ├── source_focus_scale04_*.png
│   │   └── overlay.png                ← 疊加圖，黑色=focus 不替換
│   ├── phase17_preserve/          Phase 1.7 低 attention preserve 遮罩
│   │   └── overlay.png                ← 白色=錨定區域（source token 強制保留）
│   ├── target/                    Target focus 遮罩（Phase 1.7 引導生成的 attention）
│   │   └── overlay.png
│   └── combined/                  合併 replacement 遮罩（最終 Phase 2 使用）
│       ├── combined_focus_scale*.png  ← bool 模式輸出
│       ├── combined_replace_prob_*.png ← cumulative mode 輸出（灰階機率）
│       └── overlay.png                ← 疊加圖
├── attention_cache/               ★ Attention 快取（pie_attn_cache_both=1 時建立）
│   ├── source_attention.pt
│   └── target_attention.pt
└── tokens_p2p_edit.pkl             Token + 遮罩資料
```

**遮罩視覺化規則**：

| 遮罩 | 白色 | 黑色 |
|------|------|------|
| `source_focus_*` | 背景替換區 | Source focus 區（target 自由生成）|
| `phase17_preserve/` | **低 attention 錨定區**（source image token 強制保留）| Focus 自由區 |
| `target_focus_*` | 背景替換區 | Target focus 區（改變）|
| `combined_focus_*` | 背景（source token 替換）| Union focus 區（target 自由）|
| `combined_replace_prob_*` | 高亮 = 替換機率高 | 低亮 = 替換機率低 |

---

## 與其他腳本的相容性

| 腳本 | 模型檔案 | 適用場景 |
|------|---------|---------|
| `infer.sh` | `infinity.py` | 一般圖像生成 |
| `infer_p2p.sh` | `infinity_p2p.py` | 物件替換（seed 相同結構）|
| `infer_p2p_attn.sh` | `infinity_p2p_attn.py` | 局部語義替換（無 source image）|
| `infer_p2p_edit.sh` | `infinity_p2p_edit.py` | **有 source image 的局部語義替換**（本管線）|
| `single_infer_pie_edit.sh` | `infinity_p2p_edit.py` | PIE-Bench 單一案例測試（自動讀 meta.json）|
| `batch_run_pie_edit.sh` | `infinity_p2p_edit.py` | PIE-Bench 批量評估 + schedule sweep |
