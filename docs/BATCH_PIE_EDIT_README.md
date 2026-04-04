# Batch PIE Edit

## 目的

`batch_run_pie_edit.py` / `batch_run_pie_edit.sh` 用於批量跑 `extracted_pie_bench`，並且模型只載入一次，避免每個 case 反覆載入模型造成耗時。

## 檔案位置

- `scripts/batch_run_pie_edit.sh`
- `tools/batch_run_pie_edit.py`

## 資料夾格式

預設讀取：`./outputs/outputs_loop_exp/extracted_pie_bench`

每個案例需包含：
- `image.jpg`
- `meta.json`
- `mask.png`（本批次腳本預設不使用）

`meta.json` 需有：
- `source_prompt`
- `target_prompt`

## Prompt 處理規則

每個案例在推論前會做兩步：

1. 移除 `[]` 標記（source/target 都處理）

範例：

- 原始：`a [white] raven with [green] eyes sits on a tree stump in the rain`
- 清理後：`a white raven with green eyes sits on a tree stump in the rain`

2. 找 source/target 不同詞，組成 focus words

範例：

- `source_prompt = "a black raven with red eyes sits on a tree stump in the rain"`
- `target_prompt = "a white raven with green eyes sits on a tree stump in the rain"`

會得到：
- `source_focus_words = "black red"`
- `target_focus_words = "white green"`

差異詞以空白串接（`" ".join(...)`）。

## 參數對齊

`batch_run_pie_edit.sh` 的模型與核心推論設定已對齊 `scripts/infer_p2p_edit.sh`：

- `pn=1M`
- `model_type=infinity_2b`
- `cfg=4`
- `tau=0.5`
- `image_injection_scales=2`
- `inject_weights="0.0 ... (13個)"`
- `num_full_replace_scales=2`
- `attn_threshold_percentile=80`
- `attn_block_start=2`
- `attn_block_end=-1`
- `attn_batch_idx=0`
- `p2p_token_replace_prob=0.0`
- `use_cumulative_prob_mask=1`
- `save_attn_vis=1`
- `use_normalized_attn=0`
- `seed=1`
- `phase17_fallback_replace_scales=4`
  - Single-focus fallback（只有 target focus，無 source focus）時，Phase 1.7 以 source gen token 替換前幾個 scale，讓 attention 擷取時有結構參考
  - `0` = 停用（純 free-gen）；建議值 `4`

## 執行方式

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/batch_run_pie_edit.sh
```

## 輸出結構

預設輸出到：`./outputs/outputs_loop_exp/debug`（可在腳本中修改 `output_dir`）

結構：

```text
{output_dir}/{category}/{case_id}/
  source.jpg
  target.jpg
  timing.json
  task_info.json
  case.log
  attn_masks/   # save_attn_vis=1 時
```

### 新增紀錄檔

- `task_info.json`
  - 該 case 的完整摘要：
    - status（`success` / `failed` / `skipped_existing`）
    - source/target raw prompt 與清理後 prompt
    - source/target focus words
    - `mask.png` 白/黑比例（`white_percent` / `black_percent`）
    - elapsed 秒數、輸出路徑、log 路徑
- `case.log`
  - 該 case 推論過程中的 stdout/stderr（包含 `run_one_case` 內部 print）
  - 可直接回看當時終端輸出的細節

## 常用調整

- 只跑部分 category：修改 `categories="0_xxx,1_xxx"`
- 每類只跑前 N 個：修改 `max_per_cat`
- 續跑中斷任務：保持 `skip_existing=1`
