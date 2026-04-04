# selfAttn-Edit 管線（Infinity VAR + Self-Attn Cache Ablation）

## 概述

selfAttn-Edit 是基於 P2P-Edit 的完整圖像編輯流程，新增 **self-attention cache 擷取/落盤開關**，可在不改變主流程的前提下做 ablation。

核心目標：
- 保留 source 結構（token 替換 + 可選 source image 注入）
- 讓 target prompt 在 focus 區域自由改變內容
- 可選擷取每個 scale 的 self-attn cache（用於分析/可視化）

---

## 專案結構（selfAttn-Edit 相關）

```text
InfLoop/
├── infinity/models/
│   └── infinity_selfAttn_edit.py      # selfAttn-Edit 版模型
├── tools/
│   └── run_selfAttn_edit.py           # 主程式（完整 P2P + optional self-attn cache）
├── scripts/
│   └── infer_selfAttn_edit.sh         # 啟動腳本
├── attention_map/
│   └── extractor.py                   # CrossAttentionExtractor / SelfAttentionExtractor
└── doc/
    ├── SELFATTN_EDIT_README.md        # 本文件
    └── SELFATTN_EDIT_QUICKSTART.md    # 快速上手
```

---

## 工作流程

### Phase 0：Source image 編碼（可選）
- `source_image` 有設定時：
  - 產生連續特徵（供 source 生成注入）
  - 產生各 scale 離散 token（供後續 token 替換）
- 未設定時：退回純 P2P-Attn token 來源

### Phase 1：Source 生成 + source attention 擷取
- 生成 source 圖像
- 儲存各 scale token 到 `BitwiseTokenStorage`
- 依 `source_focus_words` 計算 source focus mask

### Phase 1.6 / 1.7：Preserve + Target 引導
- 以 source 低 attention 區域建立 preserve storage
- 生成 target 引導路徑並擷取 target attention
- 得到 target focus mask

### Phase 1.9：合併遮罩 + 覆寫 token
- 合併 source/target focus mask
- 產生 replacement mask（背景替換、focus 保留）
- 若有 source image token，覆寫 storage tokens

### Phase 2：完整 Target 編輯生成
- 前幾個 scale 全域 token 替換（`num_full_replace_scales`）
- 後續 scale 以 combined mask 引導替換
- 輸出最終 `target.jpg`

### Self-Attn Cache（Ablation 開關）
- `use_self_attn_cache=1`：
  - 在完整流程中掛載 `SelfAttentionExtractor`
  - 依 `self_attn_cache_prompt` 選擇 source 或 target 路徑擷取
  - 輸出 `self_attn_cache/scale_XX.pt`（可再做熱圖可視化）
- `use_self_attn_cache=0`：
  - 不擷取 self-attn cache，主流程不變

---

## 參數說明（重點）

### Prompt / Focus
- `--source_prompt`
- `--target_prompt`
- `--source_focus_words`
- `--target_focus_words`

### P2P 核心
- `--num_full_replace_scales`：前 N 個 scale 100% source token 替換
- `--attn_threshold_percentile`：focus 區域閾值
- `--p2p_token_replace_prob`：無遮罩 fallback 機率

### Source image 注入
- `--source_image`：source 圖（可空）
- `--image_injection_scales`
- `--inject_weights`

### Self-Attn Cache Ablation
- `--use_self_attn_cache`：`0/1` 開關（完整流程內生效）
- `--self_attn_cache_prompt`：`source` 或 `target`
- `--self_attn_scale_start` / `--self_attn_scale_end`
- `--mask_image` / `--mask_threshold`：可選對齊可視化

---

## 輸出說明

預設輸出目錄（腳本）：
- `./outputs/outputs_loop_exp/selfAttn_edit_YYYYMMDD_HHMMSS/`

主要輸出：
- `source.jpg`：source 重建圖
- `target.jpg`：最終編輯圖
- `attn_masks/`：source / phase17_preserve / target / combined
- `tokens_selfAttn_edit.pkl`：token + mask 儲存

若啟用 self-attn cache（`use_self_attn_cache=1`）：
- `self_attn_cache/scale_XX.pt`：每個 scale 的 self-attn cache
- `mask_aligned/`：各 scale 對齊後 mask（有提供 `mask_image` 才會輸出）

---

## 執行方式

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/infer_selfAttn_edit.sh
```

---

## 建議的 Ablation 組合

1. 基準（不使用 self-attn cache）
- `use_self_attn_cache=0`

2. Source 路徑 cache
- `use_self_attn_cache=1`
- `self_attn_cache_prompt="source"`

3. Target 路徑 cache
- `use_self_attn_cache=1`
- `self_attn_cache_prompt="target"`

4. 範圍掃描
- 固定 prompt，掃 `self_attn_scale_start/end`
- 觀察不同 scale 對結果與熱圖的影響

---

## 注意事項

- 若磁碟空間不足，`.pt` 落盤可能失敗；建議持續輸出到 `outputs/outputs_loop_exp`。
- self-attn 擷取會增加記憶體/顯存壓力；可縮小 scale 範圍降低負擔。
- `tokens_selfAttn_edit.pkl` 是 bitwise token cache，不是 self-attn cache；可視化 self-attn 時請輸入 `self_attn_cache/`。
