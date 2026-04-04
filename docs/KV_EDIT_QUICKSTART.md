# KV-Edit 快速上手指南

## 30 秒開始

### 方式一：單一案例 KV-Edit

```bash
cd /home/avlab/Documents/InfLoop
# 修改 scripts/infer_kv_edit.sh 中的輸入路徑與 prompt
bash scripts/infer_kv_edit.sh
```

### 方式二：PIE-Bench 批量評估

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/batch_run_kv_edit.sh
```

以方式一為例，這將會：
1. 載入 source image 並進行兩種 VAE 編碼（連續特徵 + 離散 token）
2. 逐 scale **縱向交錯處理**三個 phase（而非傳統的橫向逐 phase 處理）：
   - **Source gen**：生成當前 scale 的 source token，擷取 cross-attention
   - **Dynamic mask**：從 attention peak 開始 gradient flood fill，產生空間遮罩
   - **Phase 1.7**：混入 source KV cache 結構資訊，引導 target 保留整體結構
   - **Target gen**：使用 dynamic mask + source image token 替換，生成最終結果
3. 解碼 source + target 圖片，儲存 dynamic mask 視覺化

結果存於 `--save_dir`（預設 `./outputs/kv_edit_test/`）

---

## 與 P2P-Edit 的差異

```
P2P-Edit（run_p2p_edit.py）：
  處理順序   → 水平式：source gen (scale 0→12) → phase 1.7 → target gen
  Mask 方式  → 固定 percentile 閾值
  KV cache   → 三個 phase 各自獨立，不共享結構資訊
  Phase 1.7  → preserve mask 錨定背景（離散 token）

KV-Edit（run_kv_edit.py）：
  處理順序   → 垂直式：每個 scale 依序跑 source → phase17 → target
  Mask 方式  → Dynamic mask + GT mask 引導二分法校準                ← 關鍵差異
  KV cache   → Phase17/Target 直接使用 source KV（結構資訊傳遞）   ← 關鍵差異
  Phase 1.7  → Source KV 提供結構，dynamic mask 引導局部編輯
```

---

## 檔案架構

```
InfLoop/
├── infinity/
│   ├── models/
│   │   └── infinity_p2p_edit.py              P2P-Edit 版模型
│   └── utils/
│       ├── kv_cache_manager.py               KV cache 快照管理（save/restore/blend）
│       ├── bitwise_token_storage.py           離散 token + mask 儲存
│       └── dynamic_resolution.py              Scale schedule 定義
├── attention_map/
│   └── extractor.py                          Cross-attention 擷取器
├── tools/
│   ├── run_kv_edit.py                        KV-Edit 核心引擎
│   ├── run_p2p_edit.py                       共用工具（encode, load, focus token 等）
│   └── batch_run_kv_edit.py                  PIE-Bench 批量執行器
├── scripts/
│   ├── infer_kv_edit.sh                      單一案例執行腳本
│   └── batch_run_kv_edit.sh                  PIE-Bench 批量執行腳本
└── docs/
    ├── KV_EDIT_README.md                     完整說明文件
    └── KV_EDIT_QUICKSTART.md                 本文件
```

---

## 關鍵參數說明

### `scripts/infer_kv_edit.sh`

```bash
# ── 輸入 ──
source_image="path/to/source_image.jpg"
source_prompt="a photo of a cat sitting on a couch"
target_prompt="a photo of a dog sitting on a couch"
source_focus_words="cat"             # source prompt 中要替換的物件
target_focus_words="dog"             # target prompt 中對應的新物件

# ── KV-Edit 核心參數 ──
image_injection_scales=2             # 前幾個 scale 注入 source image 連續特徵
num_full_replace_scales=2            # 前幾個 scale 100% 使用 source image token
kv_blend_scales=8                    # 前幾個 scale 的 source KV 保留在 GPU（之後 offload CPU）
gradient_threshold=0.3               # Dynamic mask 梯度擴散閾值

# ── Attention 設定 ──
attn_block_start=2                   # 從第 3 個 block 開始擷取 attention
attn_block_end=-1                    # -1 = 到最後一個 block
save_attn_vis=1                      # 儲存 dynamic mask 視覺化
```

---

## 自訂 Prompt 的修改步驟

**範例：將貓替換為狗**

```bash
# 1. 修改 scripts/infer_kv_edit.sh
source_image="./imgs/cat_on_couch.jpg"
source_prompt="a photo of a cat sitting on a couch"
target_prompt="a photo of a dog sitting on a couch"
source_focus_words="cat"
target_focus_words="dog"

# 2. 執行
bash scripts/infer_kv_edit.sh
```

**範例：移除物件**

```bash
source_image="./imgs/dog_spacesuit.jpg"
source_prompt="a dog wearing space suit"
target_prompt="a dog"
source_focus_words="space suit"
target_focus_words=""
```

---

## 參數調整速查

### 結構保留太差？

```bash
# 增加全域替換 scale 數（更多 scale 使用 source image token）
num_full_replace_scales=4

# 延長 source image 連續特徵注入範圍
image_injection_scales=4
```

### Focus 區域偏移或太小？

```bash
# 降低梯度擴散閾值（讓 mask 更容易擴散）
gradient_threshold=0.2

# 減少全域替換 scale（讓 mask 更早生效）
num_full_replace_scales=1
```

### GPU 記憶體不足？

```bash
# 減少 KV blend scale（超過此數的 scale 自動 offload KV cache 到 CPU）
kv_blend_scales=4
```

---

## 輸出檔案說明

```
outputs/kv_edit_test/
├── source.jpg                     Source 重建圖像
├── target.jpg                     編輯後圖像
└── dynamic_masks/                 Dynamic mask 視覺化（save_attn_vis=1 時輸出）
    ├── scale02_4x4.png            白色 = 編輯區域（高 attention）
    ├── scale03_4x8.png
    └── ...
```

---

## 與其他腳本的相容性

| 腳本 | 管線 | 適用場景 |
|------|------|---------|
| `infer_p2p_edit.sh` | P2P-Edit | 有 source image 的局部語義替換（水平式）|
| `infer_kv_edit.sh` | **KV-Edit** | 有 source image 的局部語義替換（垂直式 + KV 結構注入）|
| `batch_run_pie_edit.sh` | P2P-Edit 批量 | PIE-Bench 批量評估 |
| `batch_run_kv_edit.sh` | **KV-Edit 批量** | PIE-Bench 批量評估（KV-Edit 版）|
