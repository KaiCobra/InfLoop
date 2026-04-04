# KV-Edit 管線（交錯式 KV Cache 結構注入 + Dynamic Attention Mask）

## 概述

KV-Edit 是基於 P2P-Edit 的進階管線，核心創新在於**逐 scale 垂直交錯處理**與 **self-attention KV cache 結構注入**。相較於原始 P2P-Edit 的水平式處理（三個 phase 各跑完全部 scale），KV-Edit 在每個 scale 結束後立刻將 source 的 self-attention KV cache 注入到 target，讓中間 scale 即可保留結構資訊。

### 與 P2P-Edit 的核心差異

| 特性 | P2P-Edit | KV-Edit |
|------|----------|---------|
| 處理順序 | 水平式：source (s0→s12) → phase17 (s0→s12) → target (s0→s12) | 垂直式：每個 scale 依序 source → phase17 → target |
| Mask 生成 | 固定 percentile 閾值 | Dynamic mask + GT mask 引導二分法校準 |
| KV cache | 三個 phase 獨立 | Phase17/Target 直接使用 source KV（結構傳遞）|
| 結構保留 | 僅靠 token 替換 | Token 替換 + Source KV 結構傳遞（雙重保障）|

---

## 架構

### 檔案架構

```
InfLoop/
├── infinity/
│   ├── models/
│   │   ├── infinity_p2p_edit.py              P2P-Edit 模型（共用）
│   │   ├── basic.py                          基礎 block（SelfAttention, CrossAttnBlock）
│   │   └── fused_op.py                       fused_ada_layer_norm（@torch.compile）
│   └── utils/
│       ├── kv_cache_manager.py               ★ KV cache 快照管理
│       ├── bitwise_token_storage.py           離散 token + mask 儲存
│       └── dynamic_resolution.py              Scale schedule 定義
├── attention_map/
│   └── extractor.py                          Cross-attention 擷取器
├── tools/
│   ├── run_kv_edit.py                        ★ KV-Edit 核心引擎
│   ├── run_p2p_edit.py                       共用工具函式
│   └── batch_run_kv_edit.py                  PIE-Bench 批量執行器
├── scripts/
│   ├── infer_kv_edit.sh                      單一案例腳本
│   └── batch_run_kv_edit.sh                  PIE-Bench 批量腳本
└── docs/
    ├── KV_EDIT_README.md                     本文件
    └── KV_EDIT_QUICKSTART.md                 快速上手指南
```

### 核心類別

| 類別 | 檔案 | 功能 |
|------|------|------|
| `PerScaleGenerator` | `tools/run_kv_edit.py` | 封裝 Infinity 模型的 per-scale 生成邏輯，支援 token 替換、image injection |
| `KVCacheManager` | `infinity/utils/kv_cache_manager.py` | 管理三個 phase 的 KV cache 快照（save/restore/blend/offload） |
| `BitwiseTokenStorage` | `infinity/utils/bitwise_token_storage.py` | 離散 token + spatial mask 的 per-scale 儲存 |
| `CrossAttentionExtractor` | `attention_map/extractor.py` | 透過 monkey-patch 擷取 cross-attention map |

---

## 完整管線流程

### Phase 0：初始化

```
Source image → VAE encoder → ┬── 連續特徵（inject_image_features）
                             └── 離散 token（image_scale_tokens）

Source prompt → T5 encoder → source_text_cond
Target prompt → T5 encoder → target_text_cond

建立三個 PerScaleGenerator（共用同一個 Infinity 模型）：
  source_gen  ← source_text_cond
  phase17_gen ← target_text_cond
  target_gen  ← target_text_cond

建立 KVCacheManager + BitwiseTokenStorage × 3
啟用 KV caching（一次性，整個迴圈共用）
```

### Per-Scale 交錯迴圈

對每個 scale `si` (0 → total_scales-1)：

#### Step 1: Source Generation

```
restore_kv_cache('source')         ← 恢復 source 的 KV cache
register cross-attention extractor ← 準備擷取 attention

source_gen.generate_one_scale(si):
  ├── Forward through transformer blocks
  ├── Sample tokens
  ├── 前 N scale：inject source image 連續特徵到 summed_codes
  └── 儲存 source token 到 source_token_storage

save_kv_cache('source')            ← 保存 source 的 KV cache
```

#### Step 2: Dynamic Mask（si >= num_full_replace_scales 時）

```
從 cross-attention extractor 取出 source focus 的 attention map

若有 GT mask（資料集提供）→ compute_dynamic_mask_with_gt():
  1. Resize attn_map 到 GT mask 尺寸（nearest）
  2. 二分法搜尋 percentile threshold（20 次迭代）：
     - selected = attn >= threshold
     - outside = selected 且在 GT mask 外
     - inside  = selected 且在 GT mask 內
     - outside > inside → 提高 threshold; outside <= inside → 嘗試降低
  3. 用找到的 threshold 做 gradient flood fill

若無 GT mask → compute_dynamic_mask():
  1. 找到 attention map 的 peak 位置
  2. 從 peak 開始 BFS gradient flood fill
  3. 擴散條件（OR）：
     - 鄰居 attention >= threshold（絕對閾值）
     - |鄰居 - 當前| < gradient_threshold（梯度平緩 = 同區域）
  4. 嘗試多個 threshold（0.8 → 0.05），選出面積在 [min_ratio, max_ratio] 的最佳 mask

edit_mask: True = 編輯區域（高 attention to source focus）
preserve_mask = ~edit_mask: True = 背景（保留 source token）

寫入 phase17_token_storage.masks[si] ← preserve_mask
寫入 target_token_storage.masks[si]  ← preserve_mask
```

#### Step 3: Phase 1.7（Target guided with source structure）

```
restore_kv_cache('source')          ← 直接使用 source 的 KV cache（不維護獨立 KV 歷史）

phase17_gen.generate_one_scale(si):
  ├── si < num_full_replace_scales → 100% 使用 source image token
  └── si >= num_full_replace_scales → preserve_mask 引導（背景錨定 source token）

（不儲存 phase17 KV — 每個 scale 都從 source KV 重新開始）
```

#### Step 4: Target Generation

```
restore_kv_cache('source')          ← 直接使用 source 的 KV cache

如果有 target_focus_words：
  register target cross-attention extractor

target_gen.generate_one_scale(si):
  ├── si < num_full_replace_scales → 100% 使用 source image token
  └── si >= num_full_replace_scales → preserve_mask 引導（背景替換為 source token）

如果有 target attention：
  計算 target dynamic mask → union 到 dynamic_masks[si]

（不儲存 target KV — 每個 scale 都從 source KV 重新開始）
```

### Phase 3：解碼

```
source_img = source_gen.decode_image(source_state['summed_codes'])
target_img = target_gen.decode_image(target_state['summed_codes'])

儲存 source.jpg, target.jpg, dynamic_masks/
清理 KV cache、釋放 GPU 記憶體
```

---

## KV Cache 管理機制

### KVCacheManager

三個 phase 共用同一個 Infinity 模型。**只有 source 維護獨立的 KV cache 歷史**，
phase17 和 target 在每個 scale 都直接使用 source 的 KV cache（不維護自己的歷史）。

這意味著 phase17/target 的 self-attention 永遠 attend 到 source 的歷史 token，
確保結構資訊從 source 傳遞到 phase17/target。

```python
class KVCacheManager:
    save_kv_cache(model, phase_name, offload_to_cpu=False)  # 只有 source 需要 save
    restore_kv_cache(model, phase_name)                      # phase17/target 都 restore 'source'
    clear_kv_cache(model)
    save_gen_state(phase_name, ...)                          # generation 狀態仍各自獨立
    load_gen_state(phase_name) -> ScaleGenState
```

### KV Cache 流向

```
Scale si:
  Source gen → save_kv_cache('source')
  Phase17   → restore_kv_cache('source') → generate → (不 save)
  Target    → restore_kv_cache('source') → generate → (不 save)
```

Phase17/Target 在每個 scale 都「分叉」自 source 的 KV cache：
- 它們看到的是 source 在所有已完成 scale 的 KV 歷史
- 加上自身當前 scale 生成的 token（但不累積到下一個 scale）

### GPU 記憶體優化

`offload_to_cpu` 參數控制 source KV cache 快照的儲存位置：

- `si < kv_blend_scales`：快照保留在 GPU
- `si >= kv_blend_scales`：快照 offload 到 CPU，節省 GPU VRAM

`restore_kv_cache()` 自動處理 CPU → GPU 搬移。

---

## Dynamic Mask 機制

### 與 P2P-Edit 的差異

| 特性 | P2P-Edit | KV-Edit |
|------|----------|---------|
| 閾值方式 | 固定 percentile（如 top 20%） | 自適應 gradient flood fill |
| 起始點 | 全域閾值 | Attention peak |
| 擴散方式 | 無（一刀切） | BFS + 梯度條件 |
| 面積控制 | 固定比例 | 二分搜尋 + 面積約束 [2%, 85%] |
| GT mask 引導 | 無 | **有** — 二分法校準 threshold 使 mask 對齊 GT |

### Gradient Flood Fill 演算法（無 GT mask 時）

```
1. 找到 attention map 中 peak 的位置 (peak_h, peak_w)
2. 從 peak 開始 BFS
3. 對每個鄰居，擴散條件（OR）：
   - attention_score >= threshold（物件本體）
   - |neighbor_attn - current_attn| < gradient_threshold（梯度平緩 = 同區域邊界）
4. 嘗試 20 個 threshold 值（0.8 → 0.05），選出面積最合理的 mask
5. Fallback：若沒有合理 mask，使用 median threshold
```

### GT Mask 引導的二分法搜尋（有 GT mask 時）

當資料集提供 GT mask（如 PIE-Bench 的 `mask.png`），使用二分法搜尋最佳 percentile threshold：

```
1. Resize attention map 到 GT mask 尺寸（nearest interpolation）
2. 二分法搜尋（20 次迭代）：
   - mid = (low + high) / 2
   - threshold = percentile(attn, mid)
   - selected = attn >= threshold
   - outside = selected 且在 GT mask 外的像素數
   - inside  = selected 且在 GT mask 內的像素數
   - 若 outside > inside → threshold 太低，提高 percentile → low = mid
   - 若 outside <= inside → 有效，嘗試降低以擴大覆蓋 → high = mid
3. 用最佳 threshold 在目標 spatial_size 做 gradient flood fill
4. 若結果面積 < 2% → fallback 到無 GT mask 的版本
```

**效果**：GT mask 引導讓 attention-based mask 更精確地對齊物件邊界，
避免 flood fill 溢出到背景或遺漏物件邊緣。

### Token 替換三級策略

每個 scale 的 token 替換遵循三級邏輯：

| 條件 | 策略 | 說明 |
|------|------|------|
| `si < num_full_replace_scales` | 100% 替換 | 前 N scale 完全使用 source image token |
| `si >= num_full_replace_scales` 且有 mask | Mask 替換 | preserve_mask=True 的區域使用 source token |
| 無 mask 且 inject_schedule[si] == 0.0 | 全替換 | Image injection scale 也強制全替換 |

---

## PerScaleGenerator

封裝 Infinity 模型的 per-scale 生成邏輯，替代原本 `autoregressive_infer_cfg()` 的整體迴圈。

### 初始化

```python
gen = PerScaleGenerator(
    model=infinity,           # Infinity 模型（共用）
    vae=vae,                  # VAE（解碼用）
    text_cond_tuple=...,      # T5 編碼後的 text condition
    scale_schedule=...,       # [(1, h, w), ...] per-scale 解析度
    cfg_list=...,             # per-scale CFG 強度
    tau_list=...,             # per-scale 溫度
    cfg_insertion_layer=[0],  # CFG 插入位置
    vae_type=32,
    g_seed=1,
)
```

### CFG（Classifier-Free Guidance）處理

- `bs = 2 * B`：batch 包含 cond + uncond 兩份
- `cond_BD`（raw text condition）→ `shared_ada_lin` → `cond_BD_or_gss`（block forward 用）
- `sos`（= `cond_BD`）→ `get_logits()` 用（注意：不可使用 `cond_BD_or_gss`）
- CFG 插入：在指定 layer 後做 `cfg * cond + (1-cfg) * uncond`

### Source Image 注入

Source image 的連續 VAE 特徵透過 `inject_schedule` 控制注入強度：

```python
inject_schedule = [0.0, 0.0, 1.0, 1.0, ...]
# 0.0 = 100% source image 特徵（忽略 gen 的 codes）
# 1.0 = 100% gen 自由生成（不注入）
# 中間值 = 線性混合
```

注入發生在 `summed_codes` 層級：

```python
summed_codes = summed_codes * inject_w + img_feat * (1.0 - inject_w)
```

---

## 參數完整說明

### KV-Edit 核心參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--kv_blend_scales` | `8` | 前幾個 scale 的 source KV cache 保留在 GPU；超過此數 offload 到 CPU 節省 VRAM |
| `--gradient_threshold` | `0.3` | Dynamic mask 的梯度擴散閾值。越小 mask 越緊湊，越大 mask 越擴散 |
| `--num_full_replace_scales` | `2` | 前幾個 scale 100% 使用 source image token（跳過 mask） |
| `--image_injection_scales` | `2` | 前幾個 scale 注入 source image 連續特徵 |
| `--inject_weights` | `"0.0 0.0 ... 0.0"` | 各 scale 的注入權重（空格分隔）。0.0=100% image，1.0=100% free gen |

### Attention 設定

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--attn_block_start` | `2` | Cross-attention 擷取起始 block index |
| `--attn_block_end` | `-1` | Cross-attention 擷取結束 block index（-1=最後） |
| `--attn_batch_idx` | `0` | 擷取哪個 batch 的 attention（0=cond） |
| `--save_attn_vis` | `1` | 是否儲存 dynamic mask 視覺化 |

### 輸入輸出

| 參數 | 說明 |
|------|------|
| `--source_image` | Source image 路徑 |
| `--source_prompt` | Source prompt |
| `--target_prompt` | Target prompt |
| `--source_focus_words` | Source prompt 中要替換的詞彙（空格分隔） |
| `--target_focus_words` | Target prompt 中對應的新詞彙（空格分隔） |
| `--save_dir` | 輸出目錄 |

### 模型設定（與 P2P-Edit 共用）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--pn` | `1M` | 解析度（1M = 1024×1024） |
| `--model_type` | `infinity_2b` | 模型類型 |
| `--cfg` | `4` | Classifier-free guidance 強度 |
| `--tau` | `0.5` | 採樣溫度 |
| `--seed` | `1` | 隨機種子 |
| `--model_path` | `weights/infinity_2b_reg.pth` | 模型權重路徑 |
| `--vae_type` | `32` | VAE 類型 |
| `--vae_path` | `weights/infinity_vae_d32reg.pth` | VAE 權重路徑 |

---

## 批量執行（PIE-Bench）

### 資料集格式

`batch_run_kv_edit.py` 支援兩種資料集格式：

#### 格式一：PIE-Bench_v1（原始格式）

```
PIE-Bench_v1/
├── mapping_file.json          ← 全域 mapping（key=task_id, value=metadata）
├── annotation_images/         ← 各 task 的 source image
│   ├── 0_random_140/
│   │   ├── 000000000000.jpg
│   │   └── ...
│   └── ...
└── ...
```

#### 格式二：extracted_pie_bench（展開格式）

```
extracted_pie_bench/
├── 0_random_140/
│   ├── 000000000001/
│   │   ├── source.jpg
│   │   └── meta.json    ← 包含 source_prompt, target_prompt, focus_words 等
│   └── ...
└── ...
```

程式會自動偵測 `mapping_file.json` + `annotation_images/` 的存在來判斷格式。

### 批量參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--bench_dir` | — | PIE-Bench 資料集根目錄 |
| `--output_dir` | — | 輸出根目錄 |
| `--categories` | `""` | 只跑特定 category（逗號分隔，空=全部） |
| `--max_per_cat` | `-1` | 每個 category 最多跑幾個 case（-1=全部） |
| `--skip_existing` | `1` | 若 `target.jpg` 已存在就跳過（可續跑） |

---

## 管線資料流圖

```
                    Source Image
                        │
            ┌───────────┼───────────┐
            ▼                       ▼
   連續 VAE 特徵              離散 scale tokens
   (inject summed_codes)     (token storage)
            │                       │
            │                       ▼
            │              ┌─────────────────┐
            │              │ BitwiseToken     │
            │              │ Storage × 3     │
            │              │ (source/p17/tgt) │
            │              └─────────────────┘
            │                       │
            ▼                       ▼
    ┌──────────────────────────────────────────┐
    │         Per-Scale Interleaved Loop        │
    │                                          │
    │  Scale si:                               │
    │   ┌─────────────────────────────┐        │
    │   │ 1. Source Gen               │        │
    │   │    + inject image features  │        │
    │   │    + extract attention       │        │
    │   │    → save KV cache          │        │
    │   └──────────┬──────────────────┘        │
    │              │                            │
    │              ▼                            │
    │   ┌─────────────────────────────┐        │
    │   │ 2. Dynamic Mask             │        │
    │   │    有 GT mask → 二分法校準  │        │
    │   │    無 GT mask → BFS flood   │        │
    │   │    → edit_mask / preserve   │        │
    │   └──────────┬──────────────────┘        │
    │              │                            │
    │              ▼                            │
    │   ┌─────────────────────────────┐        │
    │   │ 3. Phase 1.7                │        │
    │   │    restore source KV        │        │
    │   │    preserve mask + tokens   │        │
    │   │    （不 save — 不維護 KV）  │        │
    │   └──────────┬──────────────────┘        │
    │              │                            │
    │              ▼                            │
    │   ┌─────────────────────────────┐        │
    │   │ 4. Target Gen               │        │
    │   │    restore source KV        │        │
    │   │    dynamic mask + tokens    │        │
    │   │    + extract target attn    │        │
    │   │    （不 save — 不維護 KV）  │        │
    │   └─────────────────────────────┘        │
    │                                          │
    └──────────────────────────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Decode:           │
              │  source.jpg      │
              │  target.jpg      │
              │  dynamic_masks/  │
              └──────────────────┘
```

---

## 技術細節

### torch._dynamo 相容性

KV-Edit 在啟動時設定 `torch._dynamo.config.cache_size_limit = 64`，因為三個 phase 交錯使用同一模型，動態 shape 變化頻繁，預設 cache 大小不足會導致重新編譯。

### enable_kv_caching 的放置位置

`enable_kv_caching()` 在迴圈開始前呼叫一次，而非每個 phase 切換時呼叫。原因：`sa.kv_caching(True)` 會清空 `cached_k/v`，如果在 `restore_kv_cache()` 之後呼叫，會把剛恢復的 KV cache 清除。

### CFG 與 get_logits

`get_logits()` 需要的是 raw `cond_BD`（即 `self.sos`），而非 `shared_ada_lin` 的輸出 `self.cond_BD_or_gss`。後者的 shape 為 `[bs, 1, 6, C]`，會導致 `fused_ada_layer_norm` 的 `@torch.compile` 出現 shape mismatch 錯誤。

### KV cache 維度

Infinity 的 self-attention KV cache 的 seq_len 維度取決於是否使用 flash attention：
- Flash attention：`L_dim = 1`
- 非 flash attention：`L_dim = 2`

`inject_source_kv_to_target()` 會自動偵測哪個維度不匹配來確定 `L_dim`。
