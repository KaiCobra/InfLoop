# Face-Swap Pipeline 使用說明

在 P2P-Edit 之上加一個 **face-swap** 變體：
給一組「我要的人臉」參考圖（例如 [face/smith/](../face/smith/)）+ 一個固定 prompt `T_t`，
先用 `T_t` 生成一張 base 圖 `B`，再把 prompt 中 subject token（預設 "boy"）的 T5 embedding
換成代表那張臉的向量，跑 P2P-Edit 7 階段管線把 B 上的臉換成參考身份。

支援兩種「換臉的 token 來源」：

| 模式 | 來源 | 是否需要訓練 | 換臉精準度 |
|---|---|---|---|
| **linear**（baseline） | AdaFace 512-d 人臉 embedding，repeat 4× → 2048-d，再 `λ₁·e_I + λ₂·proj(e_A)` | 否（純推論）| 偏低 — AdaFace 與 Infinity 空間不對齊，會跑出別人的臉 |
| **learned**（推薦） | Textual Inversion 學到的 `v_A ∈ R²⁰⁴⁸`，已在 Infinity 自身 embedding 空間裡 | 是（每個 identity ~5–15 分鐘）| 高 — 直接拉回 Infinity 自身空間 |

---

## 0. 前置作業

### 0.1 AdaFace HTTP server

兩種模式都需要這個 server（learned 模式仍要算 `e_B` 來做 phase 1.7 的 subtract）。
請先依 [adaface_server.md](adaface_server.md) 啟動：

```bash
nohup /media/avlab/8TB/AdaFace/server/run_server.sh --port 8000 \
      > /tmp/adaface_server.log 2>&1 &

curl -s http://127.0.0.1:8000/health
# → {"status":"ok","architecture":"ir_50","device":"cuda:0",...}
```

### 0.2 準備 source face images

把每個 identity 的多張臉照放在一個資料夾下：

```
face/
├── smith/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── jane/
│   └── ...
```

支援副檔名：`.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`。
單張也可以，但平均多張會讓 `e_A`（linear 模式）跟 `v_A`（learned 模式）更穩定。

### 0.3 模型權重

跟 P2P-Edit 共用，請確認下列檔案存在：

```
weights/
├── infinity_2b_reg.pth
├── infinity_vae_d32reg.pth
└── models--google--flan-t5-xl/snapshots/.../
```

---

## 1. 檔案架構

```
InfLoop/
├── tools/
│   ├── face_swap_utils.py                AdaFace HTTP client + token 操作 helpers
│   ├── run_pie_edit_faceSwap.py          7-phase pipeline (吃預編碼 kv)
│   ├── batch_run_pie_edit_faceSwap.py    Batch 多 identity 執行器
│   ├── optimize_face_token.py            Textual Inversion 訓練 (產 v_A.pt)
│   └── gradio_face_swap.py               Gradio demo (含 Train v_A 按鈕)
├── scripts/
│   ├── batch_run_pie_edit_faceSwap.sh    Batch face swap launcher
│   ├── optimize_face_token.sh            Textual Inversion launcher
│   └── run_gradio_face_swap.sh           Gradio demo launcher
├── face/                                 Source face images (per-identity dirs)
└── weights/identities/                   v_A.pt cache (由 optimize_face_token 產生)
    └── <identity_name>/
        ├── v_A.pt                        torch.save({"v_A":..., "v_init":..., ...})
        ├── meta.json                     iters / lr / l2_reg / final_loss / ...
        └── loss_curve.png
```

底層 `infinity/models/infinity_p2p_edit.py` **完全不動**——learned 模式直接寫入 T5 embedding，
不需要修改 transformer。

---

## 2. 用法 A — Linear 模式（不訓練、純推論）

最快的測試路徑，跟階段 1 baseline 一樣。

### 2.1 編輯設定

打開 [scripts/batch_run_pie_edit_faceSwap.sh](../scripts/batch_run_pie_edit_faceSwap.sh)：

```bash
# ── Face-Swap 設定 ──
face_root="./face"
identities=""               # csv 留空 = 跑 face/ 下全部子資料夾
prompt_t="a boy turned his head to his left over the shoulder and tilted up"
subject_word="boy"
adaface_url="http://127.0.0.1:8000"
regen_B=0                   # 1=強制重生 B
debug_face_op=0

# Phase 2 線性混合：new_e_I = lam1 * e_I + lam2 * proj(e_A)
# lam1=0, lam2=1 → 完全 replace（最常用）
# lam1=1, lam2=0 → 完全保留原 token
lam1=0.0
lam2=1.0

# 不啟用 textual inversion
use_learned_v_A=0
```

### 2.2 執行

```bash
bash scripts/batch_run_pie_edit_faceSwap.sh
```

對 `face/` 下每個子資料夾（identity）：
1. 用 `prompt_t` 生 `B.jpg`（依 `(prompt, seed)` cache）。
2. AdaFace 把這個 identity 的所有圖編成 `e_A`（多張取平均後 L2-normalize）；把 B 編成 `e_B`。
3. 三組 prompt embedding：
   - **phase 1**：`T_t` 原樣
   - **phase 1.7**：`e_I -= proj(e_B)`
   - **phase 2**：`new_e_I = λ₁·e_I + λ₂·proj(e_A)`（linear blend）
4. 跑完整 P2P-Edit 7 階段，輸出 face-swap 結果。

### 2.3 觀察

每個 identity 的輸出在 `outputs/outputs_loop_exp/face_exp2/face_swap_<...>/<id>/`：

```
smith/
├── B.jpg               # base 圖（從 prompt 生）
├── source.jpg          # phase 1 重建的 source-like（含 image injection）
├── target.jpg          # 最終 face swap 結果 ★
├── attn_masks/         # source/target/combined focus mask 視覺化
├── case.log
├── task_info.json      # identity / lam1,lam2 / e_A_norm,e_B_norm / status / ...
└── timing.json
```

如果 `target.jpg` 看起來像 `B.jpg`（沒換臉）→ 試 `lam1=-0.2, lam2=1.0`（讓原 boy 被輕微抑制）。
如果 `target.jpg` 是另一個人 → 那就是 AdaFace/Infinity 空間不對齊的本質問題，請改用 learned 模式。

---

## 3. 用法 B — Learned 模式（Textual Inversion，推薦）

> **為什麼用 `forward()` 不是 `autoregressive_infer_cfg()`**
> Textual Inversion 是 MLE：找一個 `v_A` 讓 `log P(image A | prompt with v_A)` 最大。
> 這個 likelihood 用 `Infinity.forward(text_cond, x_BLC_wo_prefix, scale_schedule)` —— 也就是 trainer 用的 teacher-forced 路徑 —— 算出來。`autoregressive_infer_cfg()` 每個 scale 做 argmax/sample 是離散的，gradient 接不過去；移除 `@torch.no_grad()` 也沒用。
> 訓練跟推論共用同一份 weights，所以用 `forward()` 學到的 `v_A` 丟回 `autoregressive_infer_cfg()` 取樣，自然會生出對應 identity。

### 3.1 訓練 v_A（每個 identity 一次，~5–15 分鐘）

打開 [scripts/optimize_face_token.sh](../scripts/optimize_face_token.sh)：

```bash
face_root="./face"
identities=""                       # 留空 = face/ 下全部
identity_cache_dir="./weights/identities"
prompt_t="a boy turned his head to his left over the shoulder and tilted up"
subject_word="boy"
steps=200                           # 100 通常已夠；複雜身份試 300–500
lr=1e-3
l2_reg=1e-4                         # ||v_A − v_init||² 正則；防 overfit
log_every=20
seed=1
regen=0                             # 1=覆寫已存在的 v_A.pt
```

執行：

```bash
bash scripts/optimize_face_token.sh
```

每個 identity 會得到：

```
weights/identities/smith/
├── v_A.pt           # {"v_A": [k,2048], "v_init": ..., "subject_token_indices": [...], ...}
├── meta.json        # iters, lr, l2_reg, init_loss, final_loss, n_images, image_paths
└── loss_curve.png
```

判斷訓練成功：
- `meta.json` 中 `final_loss << init_loss`（通常會降一個量級以上）。
- `loss_curve.png` 收斂、後段平穩。

### 3.2 推論：把 batch script 切到 learned

打開 [scripts/batch_run_pie_edit_faceSwap.sh](../scripts/batch_run_pie_edit_faceSwap.sh)，改一行：

```bash
use_learned_v_A=1
```

`identity_cache_dir` 預設 `./weights/identities` 跟 3.1 對齊，不必改。

```bash
bash scripts/batch_run_pie_edit_faceSwap.sh
```

執行時：
- 每個 identity 嘗試讀 `<cache>/<id>/v_A.pt`，找到就 phase 2 走 **learned**（直接寫入 v_A，不做 repeat-4／norm-scale）。
- 找不到會 print warning 並 fallback 到 linear（lam1/lam2 仍生效）。
- `task_info.json` 裡 `using_learned_v_A: true`、`v_A_path` 紀錄實際用的 cache。

### 3.3 訓練細節

`tools/optimize_face_token.py:optimize_v_A()` 在做這些事：

1. `kv_full = encode_prompt(T_t)`，取 boy token 對應的 slice 當 `v_init`。
2. 凍結 `infinity` + `vae` + `text_encoder`，並設 `infinity.cond_drop_rate = 0.0`
   （否則 CFG dropout 會 in-place 把 kv 換掉，gradient 全斷）。
3. 對每張 source face image 預先算 `(x_BLC_wo_prefix, gt_BL)`：
   `vae.encode_for_raw_features` → `BitwiseSelfCorrection.flip_requant`（noise 全關，乾淨 teacher-forcing）。
4. AdamW(lr) 只更新 `v_A = nn.Parameter`：
   ```python
   kv = kv_full.detach().clone()
   kv[boy_idx] = v_A
   logits = infinity((kv, lens, cu, Lmax), x_BLC, scale_schedule)
   loss   = bit_CE_with_scale_reweight(logits, gt_BL, scale_schedule, vae)
   loss  += l2_reg * ||v_A - v_init||²
   loss.backward()      # 只 v_A 接得到 grad
   opt.step()
   ```
5. 存 `v_A.pt`。

---

## 4. 用法 C — Gradio 互動式 Demo

```bash
bash scripts/run_gradio_face_swap.sh
# → 開瀏覽器 http://127.0.0.1:7860
```

UI 區塊：

| 控件 | 說明 |
|---|---|
| Prompt T_t | 預設 "a boy turned his head to his left over the shoulder and tilted up" |
| Source face image | 一張臉（linear 必填；learned 模式可空） |
| Subject word | 要操作的 token，預設 `boy` |
| Identity name | learned 模式必填；對應 `weights/identities/<name>/v_A.pt` |
| Phase 2 mode | radio：`linear` / `learned` |
| λ₁ / λ₂ | linear 模式參數 |
| Seed | 固定 → B 圖會被 cache |
| **Run face swap** | 跑單一 face swap |
| **Train v_A**（accordion） | 對上傳的 source face 跑 Textual Inversion，存到 cache |

典型流程：

1. 上傳一張臉、`Identity name` 填 `tmp_test`。
2. 預設 mode=`linear`、按 **Run face swap** 看 baseline 結果。
3. 展開 **Train v_A**：steps=150（單張 ~30s–1min on RTX 4090）→ 按 **Train v_A on this face**。
4. 訓練完看 `Training result` 的 `init_loss → final_loss` 確認下降。
5. 把 **Phase 2 mode** 切成 `learned`，再按 **Run face swap** 對比。

模型只在 server 啟動時載入一次；每次 Run 會被 lock 序列化（單 GPU）。
B 圖依 `(prompt, seed)` cache 在 `outputs/gradio_face_swap/`，prompt 不變時不會重生。

---

## 5. 實作要點與調參經驗

### 5.1 Linear 模式的 λ 直覺

- `lam1=0, lam2=1`：完全 replace（boy → AdaFace projection）
- `lam1=1, lam2=0`：完全不改（純 prompt）
- `lam1=-0.2, lam2=1.0`：把原 boy 輕微抑制再 inject 人臉，常常比純 replace 自然
- `lam2 > 1`：放大 e_A 影響（norm 已先 scale 到原 token，所以 lam2=2 是「2 倍幅度的 e_A」）

### 5.2 Textual Inversion 收斂

- `steps=100~200` 對 5 張圖通常夠；單張用 150 也能看出效果。
- `lr=1e-3` 一般沒問題；overfit 就降到 `5e-4`。
- `l2_reg=1e-4`：保 v_A 不離 boy 太遠。設 0 會 overfit；設 `1e-2` 會學不動。
- `loss_curve.png` 應該前 30 步快下、後段平穩。

### 5.3 與 Phase 1.7 的關係

phase 1.7 的「`e_I -= proj(e_B)`」邏輯沒有改——learned 模式只動 phase 2。
所以 phase 1.7 仍然依賴 AdaFace `e_B`，`AdaFace server` 在兩種模式下都需要開著。

### 5.4 task_info.json 怎麼讀

batch run 後最重要的欄位：

| 欄位 | 意義 |
|---|---|
| `using_learned_v_A` | true = 真的走 learned；false = fallback 到 linear |
| `v_A_path` | 實際讀的 cache（learned 才有） |
| `e_A_norm` / `e_B_norm` | AdaFace embedding 的 norm；應接近 1.0 |
| `subject_token_indices` | "boy" 在 prompt token 序列的位置 |
| `lam1` / `lam2` | linear 路徑使用的權重 |

### 5.5 如何客觀判斷 face-swap 成功

- 對 `target.jpg` 跟 `face/<id>/*` 都跑一次 AdaFace embed，計算 cosine
- learned 模式應該 > linear 模式 > B 對 source 的 baseline

---

## 6. Troubleshooting

| 症狀 | 可能原因 |
|---|---|
| `cannot reach server at http://127.0.0.1:8000` | AdaFace server 沒開，見 §0.1 |
| `subject_word='boy' not found in prompt` | prompt 改了但 subject_word 沒改 |
| `target.jpg` 跟 B 一樣（沒換臉） | linear 模式：試 `lam1=-0.2`；或 learned 沒收斂 |
| `target.jpg` 換成「不認識的別人」 | AdaFace/Infinity 空間不對齊；改用 learned |
| TI loss 不降 | `cond_drop_rate` 沒關（檢查 `infinity.cond_drop_rate=0.0`）；lr 太低；source images 太雜 |
| TI 後 `target.jpg` 整張壞掉 | overfit／L2 太小：把 `l2_reg` 從 `1e-4` 提到 `1e-3` 或減 steps |
| Gradio Train 按下去沒反應 | 看 console；多半是 source image 沒上傳 / identity name 沒填 |
| `weights_only=False` warning | torch 2.5 預設行為改了，無功能影響 |

---

## 7. 引用模組

- 推論主流程：[tools/run_pie_edit_faceSwap.py](../tools/run_pie_edit_faceSwap.py)
- Textual Inversion：[tools/optimize_face_token.py](../tools/optimize_face_token.py)
- Token-level embedding 操作：[tools/face_swap_utils.py](../tools/face_swap_utils.py)
- 底層 P2P-Edit：[tools/run_p2p_edit.py](../tools/run_p2p_edit.py)、[infinity/models/infinity_p2p_edit.py](../infinity/models/infinity_p2p_edit.py)
- AdaFace server API：[docs/adaface_server.md](adaface_server.md)
