# KV-Edit 在 attention 層的實際行為（基於 code）

本文件說明 KV-Edit 在 FLUX (MM-DiT) 的 attention 操作中具體做了什麼，重點回答三個問題：

1. 它是 self-attention 還是 cross-attention？
2. Q^fg 和 K/V 形狀不同怎麼對齊？
3. 在 self-attention 裡，foreground (fg) 對應 text、fg、background (bg) 三種 key 的行為各是怎麼處理？

所有引用都對到本 repo 的程式碼。

---

## 1. 是 self-attention，不是 cross-attention

FLUX 是 **MM-DiT**：text token (512) 和 image token (L) 在每個 block 裡 **concat 成一條序列做 joint self-attention**，沒有獨立的 cross-attention 層。看 [`flux/modules/layers.py:177-182`](../flux/modules/layers.py#L177-L182) 原始 `DoubleStreamBlock`：

```python
q = torch.cat((txt_q, img_q), dim=2)   # [B, H, 512+L, D]
k = torch.cat((txt_k, img_k), dim=2)
v = torch.cat((txt_v, img_v), dim=2)
attn = attention(q, k, v, pe=pe)
```

KV-Edit 不換成 cross-attention，而是把這個 joint self-attention 改成**不對稱**的 self-attention：**Q 那邊只放 text + foreground，KV 那邊放 text + (background cache + 新 foreground)**。

在語意上等價於「fg query 對著 bg+fg+text 的 KV 做 attention」，但實作上仍然是同一條 `scaled_dot_product_attention`。

---

## 2. Inversion 階段：把完整影像的 K/V 存起來

[`models/kv_edit.py:127-157`](../models/kv_edit.py#L127-L157) `Flux_kv_edit.inverse()`：傳入的 `inp["img"]` 是**完整 latent** (長度 L)。每一步 inversion 都會讓每個 block 跑「正常」的 self-attention，但會把 image 端的 K/V 存進 `info['feature']`：

[`flux/modules/layers.py:291-302`](../flux/modules/layers.py#L291-L302)（DoubleStream，inverse 分支）：

```python
feature_k_name = f"{info['t']}_{info['id']}_MB_K"
feature_v_name = f"{info['t']}_{info['id']}_MB_V"
if info['inverse']:
    info['feature'][feature_k_name] = img_k.cpu()  # [B, H, L, D]
    info['feature'][feature_v_name] = img_v.cpu()
    q = torch.cat((txt_q, img_q), dim=2)   # 完整 L，普通 self-attn
    k = torch.cat((txt_k, img_k), dim=2)
    v = torch.cat((txt_v, img_v), dim=2)
```

`SingleStreamBlock_kv` 一樣，差別只是 K/V 是從 `linear1` 出來後拆出來的 [`layers.py:351-362`](../flux/modules/layers.py#L351-L362)。每個 block、每個 timestep、每個 (K,V) 都各存一份 → cache key = `{t}_{id}_{MB|SB}_{K|V}`。

> **Inversion 時的 `attn_mask`**：開啟時 [`models/kv_edit.py:19-46`](../models/kv_edit.py#L19-L46) 會建立一個 bool mask，讓 inversion 中 **bg 只 attend bg+text、mask tokens 只 attend mask+text、互不串味**。這樣存下來的 bg K/V 不會被未來要被替換的 mask 區域汙染。

---

## 3. Denoise 階段：Q 變短，KV 維持完整 → 不對稱 attention

關鍵在 [`models/kv_edit.py:178-187`](../models/kv_edit.py#L178-L187)：

```python
mask_indices = info['mask_indices']           # 1-D LongTensor，長度 = |M|（foreground token 數）
...
inp_target["img"] = zt[:, mask_indices, ...]   # 只丟 fg tokens 進模型
```

進到 `Flux_kv.forward` 時：

- `img` shape = `[B, |M|, hidden]`，只有 fg
- `txt` shape = `[B, 512, hidden]`
- `img_ids` 仍然是**完整 L** 的位置 id（在 `prepare()` 時就建好了）

[`flux/model.py:147-151`](../flux/model.py#L147-L151) 算兩套 RoPE：

```python
ids = torch.cat((txt_ids, img_ids), dim=1)        # 512 + L
pe = self.pe_embedder(ids)                         # 給 K 用：512 + L
if not info['inverse']:
    info['pe_mask'] = torch.cat(
        (pe[:, :, :512, ...], pe[:, :, mask_indices+512, ...]), dim=2
    )                                              # 給 Q 用：512 + |M|
```

到 [`layers.py:304-315`](../flux/modules/layers.py#L304-L315) 的 denoise 分支：

```python
source_img_k = info['feature'][feature_k_name].to(img.device)   # [B, H, L, D]，整張圖的 cached K
source_img_v = info['feature'][feature_v_name].to(img.device)

mask_indices = info['mask_indices']
source_img_k[:, :, mask_indices, ...] = img_k    # 把 fg 那 |M| 格覆蓋成「現在這步剛算的 fg K」
source_img_v[:, :, mask_indices, ...] = img_v    # ↑ bg 那 (L−|M|) 格仍然是 inversion 時的 cache

q = torch.cat((txt_q, img_q), dim=2)              # [B, H, 512+|M|, D]   ← 短的
k = torch.cat((txt_k, source_img_k), dim=2)      # [B, H, 512+L,   D]   ← 長的
v = torch.cat((txt_v, source_img_v), dim=2)      # [B, H, 512+L,   D]
attn = attention(q, k, v, pe=pe, pe_q=info['pe_mask'], attention_mask=info['attention_scale'])
```

---

## 4. Q^fg 對上 KV 的形狀怎麼對齊？

**Scaled dot-product attention 本來就允許 Q 和 K 不同長度**，只要最後一維 (head_dim D) 一致：

```
softmax(Q · Kᵀ / √D) · V
[B,H,Lq,D] · [B,H,D,Lk] → [B,H,Lq,Lk] · [B,H,Lk,D] → [B,H,Lq,D]
```

這裡 `Lq = 512+|M|`、`Lk = 512+L`，輸出長度跟著 Q → `[B, H, 512+|M|, D]`。所以後面 [`layers.py:318`](../flux/modules/layers.py#L318) 才能切：

```python
txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]
#                       512                    |M|
```

**RoPE 是這裡唯一需要小心的點**：因為 Q 和 K 的位置集合不同，必須各自套自己的位置編碼。`attention()` 在 [`flux/math.py:6-16`](../flux/math.py#L6-L16) 偵測到 `pe_q` 不是 `None` 就走 `apply_rope_qk`：

- `xq` 用 `pe_q`（text + 那 |M| 個 fg 的位置）
- `xk` 用 `pe`（text + 全部 L 個 image 位置）

這樣每個 token 用的 RoPE 都對應它「在原始影像裡的真實位置」，fg query 才能和 bg key 在正確的相對位置上做點積。

---

## 5. fg query 對 text / fg / bg 的具體行為

對於每一個 fg query token，它的 attention output 是：

```
out_fg = Σ over k ∈ {text, fg, bg}  softmax(q_fg · k_*) · v_*
```

三個區段在 code 裡的來源完全不同：

| Q 對象 | K, V 來源 | 由 code 哪裡決定 | 物理意義 |
|---|---|---|---|
| **fg → text (target)** | `txt_k`, `txt_v` 是當下 denoise step 用 **target prompt** 重新跑 T5/CLIP + `txt_in` 得到的（[`gradio_kv_edit.py:197`](../gradio_kv_edit.py#L197) 用 `prompt=opts.target_prompt` 重新 `prepare()`） | `torch.cat((txt_k, source_img_k), dim=2)` 第一段 | 新內容由 **target text** 驅動 |
| **fg → fg** | `source_img_k[:,:,mask_indices,...]` **被 `img_k` 覆寫了**，所以 fg 對 fg 用的是「**這一步剛算出來的新 K/V**」，不是 cache | [`layers.py:309-310`](../flux/modules/layers.py#L309-L310) 的 in-place 覆寫 | foreground 內部的 self-organization：新內容自己跟自己對話 |
| **fg → bg** | `source_img_k` 在 `mask_indices` **以外**的位置維持 inversion 階段存下來的 K/V，**這一步不重算** | cache 從 `info['feature'][...]` 來，且只覆寫 mask 位置 | foreground 看到的是**原圖被 inversion 化的背景表徵** → 強制和原背景在語意/紋理上接得起來 |

text query (`txt_q`) 那邊也跑同一條 attention，但因為 text 端 K/V 是完整的（text + 全部 image cache），text 表徵也能繼續看到完整背景，行為跟原 FLUX 等價。

---

## 6. `attention_scale` 怎麼接到這個機制裡

[`models/kv_edit.py:48-62`](../models/kv_edit.py#L48-L62) `create_attention_scale`：

```python
attention_scale = torch.zeros(1, seq_len, dtype=torch.bfloat16, ...)  # seq_len = 512+L
attention_scale[0, background_token_indices] = scale                   # 只在 bg 那段加偏置
return attention_scale.unsqueeze(0)                                    # [1, 1, 512+L]
```

它被當作 `attn_mask` 傳進 `scaled_dot_product_attention`（PyTorch 對 float mask 是「加在 logits 上」），等於對所有 query：

```
logits[..., j] += scale   if j ∈ bg
```

→ softmax 之後 fg 對 bg 的權重變大（`scale > 0`），所以邊界更平滑、不會割裂。這是 README 說「mask 較大時調大 attn_scale」的機制原型。

---

## 7. SingleStreamBlock 行為一樣，差別只是 K/V 怎麼拆

[`layers.py:344-378`](../flux/modules/layers.py#L344-L378) 的 SingleStream：`linear1` 出來後直接從同一條 qkv 分出 `img_k = k[:, :, 512:, ...]`、`txt_k = k[:, :, :512, ...]`（FLUX 在 single-stream 是把 text+img 拼在一條 q/k/v 裡，不像 double-stream 各算各的）。Cache 與覆寫邏輯完全一樣，只是 cache key 的後綴是 `SB_K` / `SB_V`。

---

## 一句話 summary

> KV-Edit 沒改 attention 的 op，只動了它的「輸入」——把 query 縮成 fg-only、把 key/value 拼成「freshly-computed fg + cached bg」，搭配兩套 RoPE 維持位置正確性。fg query 因此能同時看到新的 target text、自己的新內容、以及原圖 inversion 出來的 bg 表徵，三者對應到 `cat` 後 K/V 序列的三個 segment。
