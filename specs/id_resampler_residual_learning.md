# Spec: IDResampler 差分學習（Residual Learning）

## 0. 為什麼要做這個（必讀）

目前 `IDResampler` 的設計：
```
out = anchor + delta(e_A, prompt_ctx)
```
其中 `anchor` 是 `"a person"` 經 T5 取出的 contextualized embedding，預先存在
`self.anchor: Parameter(n_tokens, t5_dim)`。

實測證實這條路徑**會炸**：訓練後 inference 的 diagnostic：
```
||orig_sks||  = 3.31     ← prompt 中 sks 位置 T5 contextual output
||anchor||    = 3.49
||resampler ||= 20.14    ← 6× 大
cos(resampler, anchor)   = +0.024   ← 接近正交
cos(resampler, orig_sks) = -0.025
cos(anchor,    orig_sks) = +0.017   ← anchor 跟 prompt 中真的 sks 位置就已正交
```

兩個關鍵問題：
1. Resampler 輸出方向跑出 T5 last_hidden_state 流形 → Infinity 下游崩
2. 即便方向沒飛，**`anchor`（"a person" 的 contextual emb）跟 prompt 中 `"sks"` 那位置的
   contextual emb 本來就近正交**，所以 `out = anchor + delta` 起點就不對

---

### 解法：把基底從 `anchor` 換成 `orig_sks`

```
out = orig_sks_token + g(e_A, prompt_ctx)   # g 是 Resampler 學的小殘差
```

`orig_sks_token` 就是「現在這個 prompt 跑 T5 後，sks 那幾個位置原本長的樣子」。

**為什麼這個一定比較穩**：
- 起點本來就在 valid manifold 上（T5 自己生的）
- 起點本來就跟 prompt 結構配套（cross-attn 知道怎麼處理）
- delta 一開始是 small init，所以未訓練時 `out ≈ orig_sks` → 圖跟 no_inject 幾乎一樣
- 訓練只能把 `out` 從 valid manifold 上「微微推往 ID 方向」，永遠不會跑到完全 OOD
- L2 reg 自然把殘差壓小（殘差 = 0 時 loss 最小化壓力來自 ID 對齊）

**新增規格的同時保留舊的 anchor 模式**（backward compat：舊 ckpt 還要能讀）。

---

## 1. 變更總覽

| 檔案 | 變更類型 | 重點 |
|---|---|---|
| `tools/id_resampler.py` | 改 ctor + forward 簽名 | 新增 `residual_base: 'anchor'\|'orig'`；`forward` 接收 `base_emb` |
| `tools/face_swap_utils.py` | 不動 | （`apply_resampler_to_text_features` 邏輯維持原樣） |
| `tools/train_id_resampler.py` | 改 training loop + CLI + ckpt | 抽 orig_sks → 餵 resampler；reg target 跟著 residual_base 切換 |
| `tools/infer_id_resampler.py` | 改 main flow + CLI | 抽 orig_sks → 餵 resampler；diagnostic 改用 base_emb 當對照 |
| `scripts/train_id_resampler.sh` | 加變數 | 新預設 `residual_base="orig"`、reg 加重 |
| `scripts/infer_id_resampler.sh` | 加變數 | `match_orig_norm`（之前漏接）、`residual_base` |

---

## 2. `tools/id_resampler.py`

### 2.1 `IDResampler.__init__` 新增參數

```python
def __init__(
    self,
    id_dim: int = 512,
    t5_dim: int = 2048,
    n_tokens: int = 1,
    n_id_ctx: int = 4,
    n_layers: int = 2,
    n_heads: int = 8,
    mlp_ratio: int = 4,
    use_prompt_ctx: bool = True,
    anchor_emb: Optional[torch.Tensor] = None,
    delta_init_std: float = 1e-3,
    delta_max_norm: Optional[float] = None,
    out_norm_match: str = "none",
    residual_base: str = "anchor",          # NEW
) -> None:
```

驗證：
```python
if residual_base not in ("anchor", "orig"):
    raise ValueError(f"residual_base must be 'anchor' or 'orig', got '{residual_base}'")
self.residual_base = residual_base
```

`out_norm_match` 增加合法值 `'base'`：
```python
if out_norm_match not in ("none", "anchor", "base"):
    raise ValueError(...)
```
- `'anchor'`：強制 norm 對齊 `self.anchor`（舊行為）
- `'base'`：強制 norm 對齊「實際使用的基底」（residual_base='anchor' 時等同 'anchor'，residual_base='orig' 時對齊 base_emb 的 norm）

`self.anchor` 仍照舊存在（backward compat、residual_base='anchor' 時用、`'anchor'`-mode out_norm_match 用）。即使 residual_base='orig'，`self.anchor` 仍當作 fallback / 對照儲存。

### 2.2 `IDResampler.forward` 簽名變更

```python
def forward(
    self,
    id_feat: torch.Tensor,
    prompt_ctx: Optional[torch.Tensor] = None,
    prompt_mask: Optional[torch.Tensor] = None,
    base_emb: Optional[torch.Tensor] = None,    # NEW
) -> torch.Tensor:
```

`base_emb` 形狀規範：
- `(B, n_tokens, t5_dim)` 或 `(n_tokens, t5_dim)`（若 2-d 自動 unsqueeze 到 `(1, n_tokens, t5_dim)` 並 expand 到 B）
- residual_base='orig' **必須**提供；residual_base='anchor' 時可以為 None（會被忽略）

forward 內部邏輯（取代現有 `anchor_b = self.anchor.unsqueeze(0)...`）：

```python
# ── 決定殘差基底 ──
if self.residual_base == "anchor":
    base = self.anchor.unsqueeze(0).expand(B, -1, -1).to(device)
elif self.residual_base == "orig":
    if base_emb is None:
        raise ValueError(
            "residual_base='orig' requires base_emb (shape (n_tokens, t5_dim) "
            "or (B, n_tokens, t5_dim)) at forward time"
        )
    base = base_emb.to(device).to(self.anchor.dtype)
    if base.dim() == 2:
        if base.shape != (self.n_tokens, self.t5_dim):
            raise ValueError(
                f"base_emb (2-d) shape must be ({self.n_tokens}, {self.t5_dim}), "
                f"got {tuple(base.shape)}"
            )
        base = base.unsqueeze(0).expand(B, -1, -1)
    elif base.dim() == 3:
        if base.shape[1:] != (self.n_tokens, self.t5_dim):
            raise ValueError(
                f"base_emb (3-d) last two dims must be ({self.n_tokens}, "
                f"{self.t5_dim}), got {tuple(base.shape)}"
            )
        if base.size(0) != B:
            base = base.expand(B, -1, -1)
    else:
        raise ValueError(f"base_emb must be 2-d or 3-d, got {base.dim()}-d")
else:
    raise RuntimeError(f"unreachable residual_base={self.residual_base}")

out = base + delta                                   # (B, n_tokens, t5_dim)

# ── out_norm_match：對 base 或 anchor 對齊 ──
if self.out_norm_match == "anchor":
    target_n = self.anchor.norm(p=2, dim=-1, keepdim=True).unsqueeze(0)   # (1, n_tokens, 1)
    cur_n = out.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
    out = out / cur_n * target_n.to(out.device)
elif self.out_norm_match == "base":
    target_n = base.norm(p=2, dim=-1, keepdim=True)                       # (B, n_tokens, 1)
    cur_n = out.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
    out = out / cur_n * target_n
```

> **不變的部分**：`delta_max_norm` clamp 邏輯保留在 `out = base + delta` **之前**對 `delta` 做 clamp。

### 2.3 新增 helper：`extract_orig_sks_from_text_features`

放在 `tools/id_resampler.py` 同檔，作為公開 helper。Train/Infer 都會呼叫：

```python
def extract_orig_sks_from_text_features(
    text_features: torch.Tensor,    # (1, L, t5_dim)
    sks_indices: List[int],
    n_tokens: int,
) -> torch.Tensor:
    """從 T5 last_hidden_state 中抽出 sks 對應位置的 token 當 residual base。

    Returns: (n_tokens, t5_dim) 的 tensor（同 device、同 dtype as text_features）。

    Mapping:
        n_tokens == 1                 → mean over sub-tokens                → (1, t5_dim)
        n_tokens == len(sks_indices)  → 一對一取                            → (k, t5_dim)
        其他                          → raise ValueError

    text_features 不會被修改；回傳的是 detached clone。
    """
    if not sks_indices:
        raise ValueError("sks_indices must be non-empty")
    if text_features.dim() != 3 or text_features.size(0) != 1:
        raise ValueError(f"text_features must be (1, L, D), got {tuple(text_features.shape)}")
    feats = text_features[0, sks_indices, :].detach().clone()      # (k, D)
    k = len(sks_indices)
    if n_tokens == 1:
        return feats.mean(dim=0, keepdim=True)                      # (1, D)
    if n_tokens == k:
        return feats                                                # (k, D)
    raise ValueError(
        f"n_tokens={n_tokens} 必須是 1（廣播）或 len(sks_indices)={k}（一對一）"
    )
```

---

## 3. `tools/train_id_resampler.py`

### 3.1 CLI 新增

在 Resampler 結構區塊加：

```python
parser.add_argument("--resampler_residual_base", type=str, default="orig",
                    choices=["anchor", "orig"],
                    help="Resampler 殘差的基底；'orig'=用 prompt 當下的 sks T5 output（推薦），"
                         "'anchor'=用 anchor word 的 contextual emb（舊行為）")
```

> **預設改 `'orig'`**——這是新訓練該走的路徑。`'anchor'` 留給有人想復現舊行為。

### 3.2 IDResampler 建構傳入

```python
resampler = IDResampler(
    ...
    delta_max_norm=delta_max_norm,
    out_norm_match=args.resampler_out_norm_match,
    residual_base=args.resampler_residual_base,        # NEW
)
```

### 3.3 Training loop 改寫

`tools/train_id_resampler.py:train_resampler` 內每 step 的關鍵段（找 `resampler_out = resampler(...)` 那一段）：

**Before**:
```python
resampler_out = resampler(
    id_feat=e_A.unsqueeze(0),
    prompt_ctx=text_features if resampler.use_prompt_ctx else None,
    prompt_mask=mask.bool() if resampler.use_prompt_ctx else None,
)
```

**After**:
```python
# ── 抽 orig_sks 當 residual base（即使 residual_base='anchor' 也順便算，給 reg 用）──
orig_sks_emb = extract_orig_sks_from_text_features(
    text_features, sks_idx, resampler.n_tokens
).to(device)                                                     # (n_tokens, t5_dim), fp32

base_emb = orig_sks_emb if resampler.residual_base == "orig" else None

resampler_out = resampler(
    id_feat=e_A.unsqueeze(0),
    prompt_ctx=text_features if resampler.use_prompt_ctx else None,
    prompt_mask=mask.bool() if resampler.use_prompt_ctx else None,
    base_emb=base_emb,
)
```

> 新 import：`from tools.id_resampler import IDResampler, get_anchor_t5_embedding, extract_orig_sks_from_text_features`

### 3.4 Reg target 動態化

**之前**所有 reg 都用 static `anchor_init`（在 train_resampler 開頭計算一次）。差分學習下，
`residual_base='orig'` 時 reg target 應該跟著當下 prompt 的 `orig_sks_emb` 走。

具體改寫：

```python
# 在 train_resampler 函式開頭（loop 之外）保留：
anchor_init = resampler.anchor.detach().clone()                  # (n_tokens, t5_dim)
```

每 step 計算 reg 時：

```python
# ── direction / norm regularizers ──
# 動態決定 reg target：'orig' 模式下對齊當下 orig_sks，'anchor' 模式對齊 static anchor
if resampler.residual_base == "orig":
    reg_target = orig_sks_emb.unsqueeze(0).to(resampler_out.device)   # (1, n, t5)
else:
    reg_target = anchor_init.to(resampler_out.device).unsqueeze(0)    # (1, n, t5)

if float(args.l2_anchor) > 0:
    reg_l2 = (resampler_out - reg_target).pow(2).mean()
    loss = loss + float(args.l2_anchor) * reg_l2

cos_sim_val = None
if float(args.cos_anchor) > 0:
    cs = F.cosine_similarity(resampler_out, reg_target, dim=-1)        # (1, n)
    cos_sim_val = float(cs.mean().item())
    loss = loss + float(args.cos_anchor) * (1.0 - cs.mean())

out_norm_val = None
if float(args.norm_penalty) > 0:
    target_norm = reg_target.norm(p=2, dim=-1).mean()
    out_norm = resampler_out.norm(p=2, dim=-1).mean()
    out_norm_val = float(out_norm.item())
    loss = loss + float(args.norm_penalty) * (out_norm - target_norm).pow(2)
```

> **重點**：
> - `anchor_drift` log（loop 內已有）**保留**——它仍是看 `self.anchor` Parameter 漂多少。
> - 新增一個 `base_drift` log：在 residual_base='orig' 模式下，看
>   `||resampler_out - reg_target||` 的均值（殘差大小）。

```python
# 在 (step % log_every == 0) 的分支新增：
extra = ""
if cos_sim_val is not None:
    extra += f"  cos={cos_sim_val:+.3f}"
if out_norm_val is not None:
    tgt = float(reg_target.norm(p=2, dim=-1).mean().item())
    extra += f"  out_n={out_norm_val:.3f}/tgt={tgt:.3f}"
if resampler.residual_base == "orig":
    base_drift = float((resampler_out - reg_target).norm(p=2, dim=-1).mean().item())
    extra += f"  base_drift={base_drift:.3f}"
print(f"  step {step:5d}/{int(args.steps)}  loss={loss_val:.4f}  ..."
      f"{extra}  ...")
```

> 移除原本固定 `anchor_dev = anchor_init.to(...).unsqueeze(0)` 的那行，改用上面的 `reg_target`。

### 3.5 Ckpt 存檔

`_save_ckpt` 的 `payload["config"]` 加入：

```python
"config": {
    ...
    "delta_max_norm": resampler.delta_max_norm,
    "out_norm_match": resampler.out_norm_match,
    "residual_base": resampler.residual_base,            # NEW
},
```

`payload["meta"]` 也加：
```python
"meta": {
    ...
    "residual_base": resampler.residual_base,
}
```

### 3.6 啟動 banner 加印

```python
print(f"residual_base      : {args.resampler_residual_base}")
```

---

## 4. `tools/infer_id_resampler.py`

### 4.1 CLI 新增

```python
parser.add_argument("--resampler_residual_base", type=str, default="anchor",
                    choices=["anchor", "orig"],
                    help="Resampler 殘差基底；通常從 ckpt config 讀回，CLI 只當 fallback")
parser.add_argument("--resampler_match_orig_norm", type=int, default=0, choices=[0, 1],
                    help="apply 時是否把 src 先 rescale 到 ||orig_sks||")
```

> infer 預設 `residual_base='anchor'` 維持與舊 ckpt 的相容性；新訓練的 ckpt config 會自動覆寫成 `'orig'`。

### 4.2 `build_resampler` 從 ckpt 讀 `residual_base`

在 `tools/infer_id_resampler.py:build_resampler` 內 `if isinstance(blob, dict) and "config" in blob:` 區塊加：

```python
if "residual_base" in cfg:
    residual_base = cfg["residual_base"]
else:
    residual_base = args.resampler_residual_base
```

並把 `residual_base` 傳進 IDResampler 建構：

```python
resampler = IDResampler(
    ...
    delta_max_norm=delta_max_norm,
    out_norm_match=out_norm_match,
    residual_base=residual_base,               # NEW
)
```

也在 print 加：
```python
print(f"[Resampler] ckpt config overrides: ... residual_base={residual_base}")
```

### 4.3 main flow：抽 orig_sks 並餵給 resampler

在 `tools/infer_id_resampler.py:main` 內，`resampler` baseline_mode='resampler' 分支處：

**新 import**：
```python
from tools.id_resampler import (
    IDResampler, get_anchor_t5_embedding, extract_orig_sks_from_text_features,
)
```

`mode == "resampler"` 分支（找 `with torch.no_grad(): r_out = resampler(...)` 那一段）改成：

```python
# 抽 orig_sks 給 base_emb（residual_base='anchor' 也順便算，diagnostic 會用）
orig_sks_emb = extract_orig_sks_from_text_features(
    text_features, sks_idx, resampler.n_tokens
).to(device)                                                            # (n_tokens, t5_dim)

base_emb_arg = orig_sks_emb if resampler.residual_base == "orig" else None

with torch.no_grad():
    r_out = resampler(
        id_feat=e_A_t.to(device).unsqueeze(0),
        prompt_ctx=text_features if resampler.use_prompt_ctx else None,
        prompt_mask=mask.bool() if resampler.use_prompt_ctx else None,
        base_emb=base_emb_arg,
    )                                                                   # (1, n_tokens, 2048)
replacement = r_out[0]                                                  # (n_tokens, 2048)
label = "resampler"
```

### 4.4 Diagnostic 加印 base_drift

`_print_diagnostics` 簽名擴充：

```python
def _print_diagnostics(orig_sks, anchor, replacement_vec, sks_idx,
                       label="resampler", base_emb=None):
    """印 norm + cosine 健康度指標。
    base_emb (optional, n_tokens, 2048): 若給了，額外印 cos(replacement, base) 與 base_drift。
    """
```

在原本 print 後追加：

```python
if base_emb is not None:
    b = base_emb.mean(dim=0)
    drift = float((replacement_vec - base_emb).norm(p=2, dim=-1).mean())
    print(f"[diag]   ||base||      = {float(base_emb.norm(p=2, dim=-1).mean()):.4f}")
    print(f"[diag]   cos({label}, base)     = {_cos(r, b):+.4f}")
    print(f"[diag]   ||{label}-base|| = {drift:.4f}")
```

呼叫端（`mode in ('resampler', 'anchor_only')` 那段）改：

```python
_print_diagnostics(
    orig_sks=orig_sks,
    anchor=resampler.anchor.detach().to(text_features.device),
    replacement_vec=replacement,
    sks_idx=sks_idx,
    label=label,
    base_emb=orig_sks_emb if mode == "resampler" else None,
)
```

> `anchor_only` 模式不需要 base_emb（沒跑 Resampler）。

### 4.5 Mix 寫入支援 match_orig_norm

infer 目前的 mix 寫死沒接 `match_orig_norm`。在替換 sks 位置那段：

**Before**:
```python
modified_tf[0, idx, :] = (1.0 - alpha) * text_features[0, idx, :] + alpha * src
```

**After**:
```python
if bool(args.resampler_match_orig_norm):
    # 把 src rescale 到 ||orig||（保留方向、norm 對齊）
    eps = 1e-8
    orig_norm = text_features[0, idx, :].norm(p=2).clamp_min(eps)
    src = src / src.norm(p=2).clamp_min(eps) * orig_norm
modified_tf[0, idx, :] = (1.0 - alpha) * text_features[0, idx, :] + alpha * src
```

兩個位置（n_tokens=1 廣播 / n_tokens=k 一對一）都要改。

### 4.6 啟動 banner 加印

```python
print(f"baseline_mode : {args.baseline_mode}  inject_alpha={args.inject_alpha:.3f}  "
      f"match_orig_norm={bool(args.resampler_match_orig_norm)}")
```

---

## 5. `scripts/train_id_resampler.sh`

加新變數區塊：

```bash
# ── Residual learning（差分學習）──
# residual_base:
#   orig   = Resampler 學的是「prompt 當下 sks token 的小殘差」(推薦)
#            起點在 valid manifold，不會炸
#   anchor = 用 'a person' 的 anchor 當基底（舊行為）
resampler_residual_base="orig"
```

調高預設 reg：
```bash
# direction / norm regularizers（差分學習下也要保護）
l2_anchor=1e-2
cos_anchor=1.0                            # 從 0.1 拉到 1.0：強制與 base 同向
norm_penalty=1e-1                         # 從 1e-2 拉到 1e-1：強制 norm 接近 base
```

降學習率：
```bash
lr=1e-5                                   # 從 1e-4 降到 1e-5：避免大步把殘差推飛
```

CLI 傳遞：
```bash
  --resampler_residual_base "${resampler_residual_base}" \
```
（加在現有 `--resampler_out_norm_match` 後面那一行）

啟動 banner 加：
```bash
echo " residual_base    : ${resampler_residual_base}"
```

---

## 6. `scripts/infer_id_resampler.sh`

加新變數：

```bash
# residual_base：通常 ckpt config 會覆寫，這裡設成新訓練 ckpt 預期值
resampler_residual_base="orig"
# inference 時把 src rescale 到 ||orig_sks|| 再 mix（與 residual_base 正交，可疊加）
resampler_match_orig_norm=0
```

CLI 傳遞：
```bash
  --resampler_residual_base "${resampler_residual_base}" \
  --resampler_match_orig_norm ${resampler_match_orig_norm} \
```

啟動 banner 加：
```bash
echo " residual_base : ${resampler_residual_base}  match_orig_norm=${resampler_match_orig_norm}"
```

---

## 7. Backward compatibility 檢查清單

Codex 寫完後**必須**驗證以下 backward compat 情境（用 `python -c "..."` 或寫 smoke test）：

1. **舊 ckpt（沒有 `residual_base` key）載入 inference**：
   - 應該 fallback 到 `args.resampler_residual_base`（CLI 預設 'anchor'）
   - 不應該 raise；不應該影響原本能跑的圖
2. **新 ckpt（residual_base='orig'）載入 inference**：
   - 從 ckpt config 讀回 `'orig'`
   - 自動抽 orig_sks 餵給 forward
3. **`residual_base='anchor'` 訓練（即舊行為）**：
   - 行為與本次改動前完全一致（reg target 是 static anchor_init）
   - 不需要傳 base_emb（即使傳了也忽略）
4. **`residual_base='orig'` 但忘記傳 base_emb**：
   - forward 必須 raise 明確錯誤訊息
5. **`apply_resampler_to_text_features` 不變**：
   - 它接收 resampler 已輸出的東西，不需要知道殘差是怎麼算的

---

## 8. Smoke tests（Codex 寫完跑這些）

```bash
# 1. 三檔案 compile
python -m py_compile tools/id_resampler.py tools/train_id_resampler.py tools/infer_id_resampler.py

# 2. shell syntax
bash -n scripts/train_id_resampler.sh scripts/infer_id_resampler.sh

# 3. CLI --help 含新 flag
python tools/train_id_resampler.py --help 2>&1 | grep residual_base
python tools/infer_id_resampler.py --help 2>&1 | grep -E "(residual_base|match_orig_norm)"

# 4. IDResampler unit test：兩個 mode 都能跑
python -c "
import torch
from tools.id_resampler import IDResampler, extract_orig_sks_from_text_features

# anchor mode
r1 = IDResampler(n_tokens=1, anchor_emb=torch.randn(2048)*0.5, residual_base='anchor')
out1 = r1(torch.randn(1,512), prompt_ctx=torch.randn(1,32,2048),
          prompt_mask=torch.ones(1,32,dtype=torch.bool))
print('anchor mode out shape:', out1.shape)

# orig mode
r2 = IDResampler(n_tokens=1, anchor_emb=torch.randn(2048)*0.5, residual_base='orig')
tf = torch.randn(1, 24, 2048)
base = extract_orig_sks_from_text_features(tf, [2,3,4], n_tokens=1)
print('extracted base shape:', base.shape)   # expect (1, 2048)
out2 = r2(torch.randn(1,512), prompt_ctx=torch.randn(1,32,2048),
          prompt_mask=torch.ones(1,32,dtype=torch.bool), base_emb=base)
print('orig mode out shape:', out2.shape)
print('out2 close to base?',
      ((out2[0] - base.to(out2.device)).norm() < 1.0).item(),  # delta init small → out ≈ base
)

# orig mode without base_emb should raise
try:
    r2(torch.randn(1,512), prompt_ctx=torch.randn(1,32,2048),
       prompt_mask=torch.ones(1,32,dtype=torch.bool))
    print('FAIL: should have raised')
except ValueError as e:
    print('OK: forward without base_emb raised:', e)

# n_tokens=k extract
base_k = extract_orig_sks_from_text_features(tf, [2,3,4], n_tokens=3)
print('extracted base (n_tokens=3) shape:', base_k.shape)  # expect (3, 2048)

# mismatch should raise
try:
    extract_orig_sks_from_text_features(tf, [2,3,4], n_tokens=2)
    print('FAIL: should have raised')
except ValueError as e:
    print('OK: mismatch raised:', e)
print('all OK')
"
```

期望輸出：
```
anchor mode out shape: torch.Size([1, 1, 2048])
extracted base shape: torch.Size([1, 2048])
orig mode out shape: torch.Size([1, 1, 2048])
out2 close to base? True            ← 未訓練時 delta 小，out ≈ base
OK: forward without base_emb raised: ...
extracted base (n_tokens=3) shape: torch.Size([3, 2048])
OK: mismatch raised: ...
all OK
```

---

## 9. 預期效果（Codex 改完後 user 跑來看）

### 用舊 ckpt（residual_base='anchor'）做 inference
- 行為與改動前完全一致（除了 diagnostic 多印一行 `||base||` / `cos(resampler, base)`）
- 圖一樣會崩（這是預期；舊 ckpt 就是 manifold drift 的）

### 重訓（residual_base='orig'，新預設）後 inference
重點 diagnostic 應該變成：
```
||orig_sks||  ≈ 3.3
||base||      ≈ 3.3      ← 與 orig_sks 一致（就是 orig_sks）
||resampler|| ≈ 3.3-4.0  ← 跟 base 同數量級（不再是 20）
cos(resampler, base) > 0.7    ← 與 base 同向（殘差小）
||resampler-base|| < 1.5      ← 殘差絕對值小
```

對應的圖：
- α=0：等同 no_inject（與之前一致）
- α=0.3：輕度 ID，背景結構正常
- α=1.0：完整 ID，背景結構**仍正常**（不再崩）

### 重訓中應該看到的 log
```
step    20/2000  loss=0.6512  ... cos=+0.123 out_n=3.412/tgt=3.317  base_drift=2.142  ...
step   500/2000  loss=0.5234  ... cos=+0.687 out_n=3.301/tgt=3.295  base_drift=0.873  ...
step  2000/2000  loss=0.4612  ... cos=+0.914 out_n=3.298/tgt=3.300  base_drift=0.412  ...
```

`cos` 從低慢慢爬到 0.9 以上、`base_drift` 慢慢縮小、`out_n` 始終貼近 `tgt` ≈ 3.3。

如果訓練完 `cos < 0.5` 或 `base_drift > 2.0`，把 `cos_anchor` 再加倍、`norm_penalty` 再加倍。

---

## 10. 不要做的事

- **不要動** `tools/face_swap_utils.py`：它的 `apply_resampler_to_text_features` 只負責「拿 Resampler 已輸出的東西寫到 sks 位置」，不知道也不該知道 residual 是怎麼算的。
- **不要刪** `IDResampler.anchor`：要保留作為 backward compat、residual_base='anchor' 模式、`out_norm_match='anchor'` 模式都會用。
- **不要改** ckpt loading 的 `state_dict` 部分：`anchor` 仍是 Parameter，state_dict 會自動處理。
- **不要把 reg target 寫死成 anchor_init**：`residual_base='orig'` 必須用 dynamic `orig_sks_emb`。
- **不要在 `apply_resampler_to_text_features` 中重新計算 norm match**：infer 端的 `--resampler_match_orig_norm` 是另一層獨立控制，跟 forward 內 `out_norm_match` 是兩件事，不要混淆。
- **不要在 inference 時把 prompt_ctx 餵成 modified_tf**：應該餵**原始** `text_features`（pre-replacement 的乾淨版）。Resampler 看的是 prompt 自然 context 來判斷該注入什麼 ID 樣態。

---

## 11. Files to deliver

```
tools/id_resampler.py            (modified: ctor + forward + new helper)
tools/train_id_resampler.py      (modified: training loop + reg + CLI + ckpt)
tools/infer_id_resampler.py      (modified: main flow + CLI + diagnostic + match_orig_norm wiring)
scripts/train_id_resampler.sh    (modified: new vars + adjusted defaults)
scripts/infer_id_resampler.sh    (modified: new vars)
```

不新增任何檔案；不動 `tools/face_swap_utils.py`、`tools/optimize_face_token.py`、其他 P2P-Edit 相關檔案。
