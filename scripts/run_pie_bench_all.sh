#!/bin/bash
# ==============================================================================
# run_pie_bench_all.sh — PIE-Bench 五配置批次實驗 + 評估
#
# 配置：
#   1. P2P num_source_scales=1
#   2. P2P num_source_scales=2
#   3. P2P num_source_scales=4
#   4. P2P num_source_scales=6
#   5. P2P-Attn (attention-guided, num_full_replace=4, threshold=75)
#
# 使用方式：
#   bash scripts/run_pie_bench_all.sh validate   # 快速驗證 (每 cat 2 cases)
#   bash scripts/run_pie_bench_all.sh full        # 全部 700 cases
# ==============================================================================

MODE="${1:-validate}"

echo "================================================================"
echo " PIE-Bench 五配置批次實驗"
echo " Mode: ${MODE}"
echo "================================================================"

# ── 路徑設定 ──
BENCH_DIR="outputs/outputs_loop_exp/extracted_pie_bench"
RESULT_BASE="outputs/outputs_loop_exp"
EVAL_BASE="outputs/eval_pie"

# ── 模型設定 ──
MODEL_PATH="weights/infinity_2b_reg.pth"
VAE_PATH="weights/infinity_vae_d32_reg.pth"
T5_PATH="weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001"

# ── 生成設定 ──
CFG=4.0
TAU=0.5
SEED=0
P2P_REPLACE_PROB=0.5

# ── Attn 設定 ──
ATTN_FULL_REPLACE=4
ATTN_THRESHOLD=75.0

# ── 執行 ──
python3 tools/run_pie_bench_batch.py \
  --mode "${MODE}" \
  --bench_dir "${BENCH_DIR}" \
  --result_base "${RESULT_BASE}" \
  --eval_base "${EVAL_BASE}" \
  --model_path "${MODEL_PATH}" \
  --vae_path "${VAE_PATH}" \
  --text_encoder_ckpt "${T5_PATH}" \
  --pn 1M \
  --model_type infinity_2b \
  --vae_type 32 \
  --text_channels 2048 \
  --rope2d_each_sa_layer 1 \
  --rope2d_normalized_by_hw 2 \
  --use_bit_label 1 \
  --cfg ${CFG} \
  --tau ${TAU} \
  --seed ${SEED} \
  --p2p_token_replace_prob ${P2P_REPLACE_PROB} \
  --attn_num_full_replace_scales ${ATTN_FULL_REPLACE} \
  --attn_threshold_percentile ${ATTN_THRESHOLD} \
  --configs all

echo ""
echo "================================================================"
echo " 完成！結果位於："
echo "  生成結果: ${RESULT_BASE}/p2p_s{1,2,4,6}/ 和 ${RESULT_BASE}/p2p_attn/"
echo "  評估結果: ${EVAL_BASE}/p2p_s{1,2,4,6}/ 和 ${EVAL_BASE}/p2p_attn/"
echo "================================================================"
