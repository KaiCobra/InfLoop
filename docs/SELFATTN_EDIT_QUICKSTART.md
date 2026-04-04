# selfAttn-Edit 快速上手

## 30 秒開始

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/infer_selfAttn_edit.sh
```

這會執行完整 selfAttn-Edit 管線（Source → Mask → Target）。

---

## 最常用切換

編輯 `scripts/infer_selfAttn_edit.sh`：

### A. 完整 P2P（不擷取 self-attn cache）
```bash
use_self_attn_cache=0
```

### B. 完整 P2P + 擷取 source self-attn cache
```bash
use_self_attn_cache=1
self_attn_cache_prompt="source"
self_attn_scale_start=0
self_attn_scale_end=-1
```

### C. 完整 P2P + 擷取 target self-attn cache
```bash
use_self_attn_cache=1
self_attn_cache_prompt="target"
```

---

## 最小必要參數

- `source_prompt` / `target_prompt`
- `source_focus_words` / `target_focus_words`
- `source_image`（可空）
- `num_full_replace_scales`
- `attn_threshold_percentile`

---

## 輸出位置

每次執行會建立：
- `./outputs/outputs_loop_exp/selfAttn_edit_YYYYMMDD_HHMMSS/`

重要檔案：
- `source.jpg`
- `target.jpg`
- `attn_masks/`
- `tokens_selfAttn_edit.pkl`
- `self_attn_cache/`（只有 `use_self_attn_cache=1` 才有）

---

## self-attn 快速可視化

```bash
python3 tools/visualize_self_attn_cache.py \
  --input outputs/outputs_loop_exp/selfAttn_edit_*/self_attn_cache
```

若要指定裝置：
```bash
python3 tools/visualize_self_attn_cache.py \
  --input outputs/outputs_loop_exp/selfAttn_edit_*/self_attn_cache \
  --device cuda
```

---

## 常見問題

1. 沒有 `self_attn_cache/`
- 檢查 `use_self_attn_cache=1`

2. 顯存不足或過慢
- 降低 `self_attn_scale_start/end` 範圍
- 暫時關閉 cache：`use_self_attn_cache=0`

3. 想只看 target 改動區域
- 適度降低 `num_full_replace_scales`
- 調整 `attn_threshold_percentile`
