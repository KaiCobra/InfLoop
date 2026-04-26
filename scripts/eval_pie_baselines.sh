#!/bin/bash
# ==============================================================================
# eval_pie_baselines.sh — 批次評估 eval_pie_ours/output 內的所有 baseline 方法
#
# 支援兩種目錄格式：
#
#   Format A（舊）：{method}/samples/{case_id}/edited.jpg
#     → 自動建立 symlink adapter 結構供 eval 使用
#
#   Format B（新）：{method}/images/{category}/{case_id}/target.jpg
#     → 直接傳入 eval，不需轉換
#
# 判斷邏輯：若 {method}/images/ 存在則為 Format B，否則為 Format A
#
# 輸出：
#   outputs/eval_pie_baselines/results/{method}/per_case.csv
#   outputs/eval_pie_baselines/results/{method}/summary.json
#   outputs/eval_pie_baselines/logs/{method}.txt
# ==============================================================================

set -euo pipefail

# 強制使用本地快取，避免 HuggingFace 網路逾時
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── 選擇要評估的方法 ──
# 用法：bash eval_pie_baselines.sh [method1] [method2] ...
#   無引數 → 互動選單
#   引數   → 直接指定方法名稱（空白分隔，可多選）
METHODS_DIR_TMP="outputs/outputs_loop_exp/eval_pie_ours/output"
SELECTED_METHODS=()

if [ $# -gt 0 ]; then
    # 命令列直接指定
    SELECTED_METHODS=("$@")
    echo "[Select] 命令列指定方法：${SELECTED_METHODS[*]}"
else
    # 互動選單
    mapfile -t ALL_METHODS < <(ls -1 "${METHODS_DIR_TMP}/")
    echo ""
    echo "══════════════════════════════════════════"
    echo "  請選擇要評估的方法（輸入序號，多選用空白分隔）"
    echo "  0 = 全部評估"
    echo "══════════════════════════════════════════"
    for i in "${!ALL_METHODS[@]}"; do
        printf "  [%2d] %s\n" "$((i+1))" "${ALL_METHODS[$i]}"
    done
    echo "══════════════════════════════════════════"
    read -rp "  請輸入序號（例如：1 3 7）或 0 全選：" -a USER_CHOICES

    if [ "${#USER_CHOICES[@]}" -eq 0 ]; then
        echo "[Abort] 未選擇任何方法，離開。"
        exit 0
    fi

    # 解析選擇
    for choice in "${USER_CHOICES[@]}"; do
        if [ "$choice" -eq 0 ] 2>/dev/null; then
            SELECTED_METHODS=("${ALL_METHODS[@]}")
            echo "[Select] 全部 ${#ALL_METHODS[@]} 個方法"
            break
        elif [ "$choice" -ge 1 ] && [ "$choice" -le "${#ALL_METHODS[@]}" ] 2>/dev/null; then
            SELECTED_METHODS+=("${ALL_METHODS[$((choice-1))]}")
        else
            echo "[Warn] 無效序號：${choice}，略過"
        fi
    done

    if [ "${#SELECTED_METHODS[@]}" -eq 0 ]; then
        echo "[Abort] 未選擇有效方法，離開。"
        exit 0
    fi
    echo "[Select] 已選擇：${SELECTED_METHODS[*]}"
fi
echo ""

BENCH_DIR="./outputs/outputs_loop_exp/PIE-Bench_v1-20260314T125823Z-3-001/PIE-Bench_v1/"
METHODS_DIR="outputs/outputs_loop_exp/eval_pie_ours/output"
ADAPTED_DIR="outputs/outputs_loop_exp/eval_pie_ours/adapted"
RESULTS_DIR="outputs/eval_pie_baselines/results"
LOG_DIR="outputs/eval_pie_baselines/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "================================================================"
echo " PIE-Bench Baseline 批次評估"
echo " methods_dir : ${METHODS_DIR}"
echo " bench_dir   : ${BENCH_DIR}"
echo " results_dir : ${RESULTS_DIR}"
echo " logs_dir    : ${LOG_DIR}"
echo "================================================================"
echo ""

# ── 建立 case_id → category 對照表（Format A 用）──
python3 - <<'PYEOF'
import json, os

mapping_file = "./outputs/outputs_loop_exp/PIE-Bench_v1-20260314T125823Z-3-001/PIE-Bench_v1/mapping_file.json"
with open(mapping_file) as f:
    mapping = json.load(f)
with open("/tmp/_pie_case_to_cat.txt", "w") as f:
    for case_id, info in mapping.items():
        cat = info["image_path"].split("/")[0]
        f.write(f"{case_id}\t{cat}\n")
print(f"[Setup] mapping_file.json 載入完成，共 {len(mapping)} 個 case")
PYEOF

# ── 逐一評估每個方法 ──
for method_dir in "${METHODS_DIR}"/*/; do
    method=$(basename "$method_dir")

    # 過濾：只處理已選擇的方法
    selected=0
    for sel in "${SELECTED_METHODS[@]}"; do
        if [ "$sel" = "$method" ]; then
            selected=1
            break
        fi
    done
    [ "$selected" -eq 0 ] && continue

    echo "────────────────────────────────────────────────────────────"

    # ── 判斷格式 ──
    if [ -d "${method_dir}images" ]; then
        # Format B：{method}/images/{category}/{case_id}/target.jpg
        result_dir="${method_dir}images"
        case_count=$(ls "${method_dir}images/"*/ 2>/dev/null | grep -c "^" || true)
        echo "[Method] ${method}  [Format B: images/{cat}/{case}/target.jpg]"

    elif [ -d "${method_dir}samples" ]; then
        # Format A：{method}/samples/{case_id}/edited.jpg → 需要建 symlink
        samples_dir="${method_dir}samples"
        case_count=$(ls "$samples_dir" | wc -l)
        echo "[Method] ${method}  [Format A: samples/{case}/edited.jpg]  (${case_count} cases)"

        adapted="${ADAPTED_DIR}/${method}"
        rm -rf "$adapted"

        python3 - "$samples_dir" "$adapted" <<'PYEOF'
import sys, os

samples_dir = sys.argv[1]
adapted_dir = sys.argv[2]

case_to_cat = {}
with open("/tmp/_pie_case_to_cat.txt") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            case_to_cat[parts[0]] = parts[1]

count = 0
skipped = 0
for case_id in sorted(os.listdir(samples_dir)):
    case_path = os.path.join(samples_dir, case_id)
    if not os.path.isdir(case_path):
        continue

    cat = case_to_cat.get(case_id)
    if cat is None:
        skipped += 1
        continue

    edited = os.path.join(case_path, "edited.jpg")
    if not os.path.exists(edited):
        skipped += 1
        continue

    target_dir = os.path.join(adapted_dir, cat, case_id)
    os.makedirs(target_dir, exist_ok=True)
    target_link = os.path.join(target_dir, "target.jpg")
    if os.path.lexists(target_link):
        os.remove(target_link)
    os.symlink(os.path.abspath(edited), target_link)
    count += 1

print(f"[Adapt] symlink 建立：{count} 個，跳過：{skipped} 個")
PYEOF
        result_dir="${adapted}"

    else
        # Format C：{method}/{category}/{case_id}/target.jpg（category 直接在 method 下）
        # 判斷：method_dir 下的第一層子目錄，裡面有 case 子目錄且含 target.jpg
        first_sub=$(ls -d "${method_dir}"*/ 2>/dev/null | head -1 || true)
        if [ -n "$first_sub" ]; then
            first_case=$(ls -d "${first_sub}"*/ 2>/dev/null | head -1 || true)
            if [ -n "$first_case" ] && [ -f "${first_case}target.jpg" ]; then
                result_dir="${method_dir%/}"
                cat_count=$(ls -d "${method_dir}"*/ 2>/dev/null | wc -l)
                echo "[Method] ${method}  [Format C: {cat}/{case}/target.jpg]  (${cat_count} categories)"
            else
                echo "[Skip] ${method} — 無法識別目錄格式"
                continue
            fi
        else
            echo "[Skip] ${method} — 找不到任何子目錄"
            continue
        fi
    fi

    echo "────────────────────────────────────────────────────────────"

    # ── 執行評估 ──
    output_csv="${RESULTS_DIR}/${method}/per_case.csv"
    summary_json="${RESULTS_DIR}/${method}/summary.json"
    log_file="${LOG_DIR}/${method}.txt"
    mkdir -p "${RESULTS_DIR}/${method}"

    echo "[Eval] 開始評估 → ${log_file}"

    python3 tools/eval_pie_results.py \
        --bench_dir    "${BENCH_DIR}" \
        --result_dir   "${result_dir}" \
        --output_csv   "${output_csv}" \
        --summary_json "${summary_json}" \
        --skip_missing 1 \
        2>&1 | tee "${log_file}"

    exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "[Done] ${method} 評估完成  ✓"
    else
        echo "[Error] ${method} 評估失敗，exit code=${exit_code}"
    fi
    echo ""
done

echo "================================================================"
echo "[All Done] 所有方法評估完成"
echo "  結果 CSV  : ${RESULTS_DIR}/"
echo "  log txt   : ${LOG_DIR}/"
echo "================================================================"
