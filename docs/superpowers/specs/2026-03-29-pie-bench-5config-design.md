# PIE-Bench 五配置批次實驗設計

## 目標
在 PIE-Bench 700 cases 上跑 5 種 P2P 配置，並對每種配置做定量評估。

## 5 種配置

| # | 腳本 | 關鍵參數 | 輸出名稱 |
|---|------|---------|---------|
| 1 | run_p2p.py | num_source_scales=1 | p2p_s1 |
| 2 | run_p2p.py | num_source_scales=2 | p2p_s2 |
| 3 | run_p2p.py | num_source_scales=4 | p2p_s4 |
| 4 | run_p2p.py | num_source_scales=6 | p2p_s6 |
| 5 | run_p2p_attn.py | num_full_replace=4, threshold=75 | p2p_attn |

## 共用參數
- cfg=4, tau=0.5, seed=0, pn=1M, model_type=infinity_2b
- p2p_token_replace_prob=0.5
- 模型: weights/infinity_2b_reg.pth
- VAE: weights/infinity_vae_d32_reg.pth (vae_type=32)
- T5: weights/models--google--flan-t5-xl/...

## 流程
1. 驗證階段：每 category 2 cases (20 total x 5 configs = 100 推論)
2. 全量階段：700 cases x 5 configs = 3500 推論
3. 評估：每種配置獨立 eval

## Source Image 策略
使用 prompt 生成 source（非真實圖片）。eval 時用 `--source_from_result` 讓 eval
讀取 result_dir 裡的 source.jpg 而非 bench_dir 的 image.jpg。

## 輸出結構
```
outputs/outputs_loop_exp/
  pie_bench_results_p2p_s1/{cat}/{case_id}/source.jpg, target.jpg
  pie_bench_results_p2p_s2/...
  pie_bench_results_p2p_s4/...
  pie_bench_results_p2p_s6/...
  pie_bench_results_p2p_attn/...

outputs/eval_pie/
  p2p_s1/per_case.csv, summary.json
  p2p_s2/...
  p2p_s4/...
  p2p_s6/...
  p2p_attn/...
```

## 實作元件
1. `tools/run_pie_bench_batch.py` — 主批次腳本，載入模型一次，遍歷所有 case
2. `scripts/run_pie_bench_all.sh` — shell wrapper（設路徑+參數，呼叫 Python）
3. `eval_pie_results.py` 新增 `--source_from_result` flag
