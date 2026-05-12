# SAM3 Face Identity Adapter for Infinity

## 任務大綱與時間

**任務目標**  
將目前以 AdaFace 向量與 T5 token replacement 為主的 IDResampler 路線，升級成以 **SAM3 image feature pyramid** 為輸入的 visual identity conditioning。目標是讓 source face 的 identity 能泛化到不同 pose / structure 的 target prompt，而不是把 source image 的空間結構硬塞進 Infinity。

**核心假設**  
直接把 source face 的 spatial feature 加到每個 scale input，可能會把 source pose / geometry 一起帶入 target generation。因此 v1 不以 `last_stage = last_stage + spatial_map` 作為主路徑，而是先把 SAM3 pyramid 經 face parsing masks 聚合成 **pose-agnostic region identity tokens**，再用 visual cross-attention adapter 注入 Infinity。

**預估時間**

| 階段 | 時間 | 產出 |
|---|---:|---|
| Phase 0: 介面確認 | 0.5-1 天 | 確認 SAM3 feature extractor 輸出格式、face-parsing mask 類別、Infinity block hook 點 |
| Phase 1: Prototype adapter | 2-3 天 | `SAM3FacePyramidAdapter`、region token pooling、zero-init visual cross-attn branch |
| Phase 2: Training pipeline | 2-4 天 | freeze SAM3/Infinity，訓練 adapter；支援 scale 5-13、region-mask loss、checkpoint |
| Phase 3: Inference pipeline | 1-2 天 | prompt T2I + visual ID adapter inference；支援逐 checkpoint 比較 |
| Phase 4: Debug / ablation | 3-5 天 | scale/block/gate/LoRA ablation，觀察 identity、pose preservation、prompt following |

第一個可跑版本預估 **5-8 天**。完整穩定版本視 SAM3 feature 品質與 loss 設計，可能需要 **2 週以上**。

## 背景與問題

目前 IDResampler 嘗試將 face ID 映射成 T5 output token，並替換 prompt 中 `sks` 的 contextual embedding。即使加入 residual learning、mid-scale loss，仍可能遇到：

- T5 token replacement 容量太小，難以表達 face identity。
- learnable token 容易離開 Infinity 熟悉的 text embedding manifold。
- 一個 token 同時承擔 identity、pose、texture、layout，任務過度糾纏。
- AdaFace 單一 global vector 缺少眼、鼻、嘴、臉型等局部結構資訊。

新的方向是把 face identity 視為 **visual conditioning**，而不是 text embedding replacement。

## 輸入與外部模組

### SAM3 Feature Extractor

輸入永遠是一個 source face image tensor。

```python
image: Tensor[B, 3, H, W]
```

輸出為多尺度 pyramid：

```python
pyramid: List[Tensor[B, 1024, h_i, w_i]]
```

SAM3 frozen，不參與訓練。1024-d feature 先透過 adapter 投影到 Infinity hidden dim 或 adapter hidden dim。

### Face Parsing

使用 `yakhyo/face-parsing` 取得 face segmentation masks。重點區域：

- skin / face
- left_eye, right_eye
- left_eyebrow, right_eyebrow
- nose
- mouth / upper_lip / lower_lip
- hair optional

mask 主要用途：

- 從 SAM3 feature 中做 region-aware pooling。
- 訓練 loss 只集中在 face identity relevant 區域，降低背景與 pose overfit。

## Adapter 設計

### 不建議主路徑：Spatial Residual Injection

直接對每個 Infinity scale 做：

```python
last_stage = last_stage + visual_adapter(scale=si, sam_pyramid=pyramid)
```

風險是 source face angle 與 target prompt face angle 不一致時，spatial feature 會把 source pose / geometry 一起注入。例如 source 是正臉，target prompt 是側臉，模型可能被迫回到正臉。

這個路徑可以保留為 ablation，但不作為 v1 主線。

### 推薦主路徑：Region Identity Tokens + Visual Cross-Attention

SAM3 pyramid 先轉成 pose-agnostic region tokens：

```text
SAM3 pyramid
  -> per-scale projection
  -> face parsing mask resize
  -> masked average pooling per face part
  -> region tokens
  -> small transformer adapter
  -> id_tokens
```

每個 identity token 代表一個 face part 或 face-part relation，而不是 source image 上的固定位置。

範例 token set：

```text
[skin, left_eye, right_eye, left_brow, right_brow, nose, mouth, upper_lip, lower_lip, global_face]
```

adapter 內部：

```text
region tokens + scale embedding + learnable id queries
  -> Transformer encoder/decoder
  -> id_tokens: Tensor[B, N_id, C_infinity]
```

建議初始設定：

- `N_id = 8-16`
- adapter hidden dim `1024` 或 `2048`
- transformer layers `2-4`
- heads `8`
- output dim 對齊 Infinity `C = 2048`

## 接到 Infinity 的方式

### v1: 新增 Visual Cross-Attention Branch

在 Infinity selected blocks 中加入獨立 visual cross-attention：

```python
x = x + text_cross_attn(...)
x = x + gate[scale, block] * visual_cross_attn(norm(x), id_tokens)
x = ffn(...)
```

重點：

- 不把 visual tokens 混進 T5 `ca_kv`，避免污染 text branch。
- 原始 Infinity weights frozen。
- `gate` zero-init，初始行為完全等於原本 Infinity。
- 只訓練 visual adapter 與 visual cross-attn branch。

建議接入位置：

- scale：先開 `5..13`，對應 1-based 第 6 到第 14 個概念若用 0-based 則為 `4..12` 或依實際 schedule 調整。
- block：先開 middle-to-late blocks，例如 `16..31`。
- 第一版不要每個 block 都接，避免過度干擾生成 manifold。

### v2: FFN / Cross-Attn LoRA

若 v1 identity strength 不夠，再加 PEFT：

- visual branch LoRA on cross-attn projections
- FFN LoRA on selected late blocks

建議：

- LoRA rank `4` 或 `8`
- zero-init LoRA output
- 只開 scales `5..13`
- 只開 blocks `20..31` 作為第一輪

不建議直接 full finetune Infinity cross-attn + FFN，資料量不足時容易破壞 prompt following 與生成分布。

## Loss 設計

### Region-Aware Bitwise CE

使用 teacher-forced Infinity bitwise CE，但只在 face parsing mask 對應區域計 loss。

流程：

```text
source image
  -> VAE encode / BSC 得到 per-scale GT tokens
  -> face parsing mask downsample 到每個 Infinity scale
  -> selected face region 上計 CE
```

loss：

```python
loss = sum_scale sum_region w_region * CE(scale, region_mask)
```

建議 region weights：

| Region | Weight | 理由 |
|---|---:|---|
| eyes / brows | high | ID 辨識強 |
| nose | high | ID 辨識強 |
| mouth / lips | high | ID 辨識強 |
| skin / face area | medium | 臉型與膚色，但易受光照影響 |
| hair | optional | 可幫助 ID，但容易學到 source-specific hairstyle |
| background / neck / clothes | zero | 避免學 pose/background |

### Regularization

需要保護 visual adapter 不把 source spatial layout 學死：

- gate L2 / small-init
- id token norm regularization
- dropout region tokens
- random face crop / color jitter
- optional pose augmentation

### Training Objective

第一版：

```text
freeze SAM3
freeze Infinity
freeze VAE
train:
  - SAM3FacePyramidAdapter
  - visual cross-attn branch
  - zero-init gates
loss:
  - region-aware bitwise CE on face masks
scales:
  - 5..13
```

## 實作計劃

### Phase 0: 準備介面

- 新增 SAM3 feature extractor wrapper。
- 新增 face-parsing wrapper，輸出固定 class masks。
- 定義 scale mapping：SAM3 pyramid level 對應 Infinity scale。
- 寫 debug dump：儲存 masks、downsample masks、region token norm。

### Phase 1: Adapter Module

新增模組：

```text
tools/sam3_face_features.py
tools/face_parsing_masks.py
infinity/models/face_identity_adapter.py
```

核心 class：

```python
class SAM3FacePyramidAdapter(nn.Module):
    def forward(self, pyramid, face_masks, scale_idx) -> Tensor[B, N_id, C]
```

### Phase 2: Infinity Hook

新增可選參數，不影響原本 pipeline：

```python
visual_id_tokens_by_scale: Optional[Dict[int, Tensor]]
visual_adapter: Optional[nn.Module]
visual_adapter_scales: Tuple[int, int]
visual_adapter_blocks: List[int]
```

在 selected `CrossAttnBlock` 後加入 visual branch。

### Phase 3: Training

新增訓練腳本，不改舊 IDResampler：

```text
tools/train_sam3_face_adapter.py
scripts/train_sam3_face_adapter.sh
```

支援：

- 每 N step checkpoint
- region loss log
- per-scale loss log
- gate norm log
- id token norm log

### Phase 4: Inference

新增：

```text
tools/infer_sam3_face_adapter.py
scripts/infer_sam3_face_adapter.sh
```

輸入：

- prompt
- source face image
- adapter checkpoint

輸出：

- generated images
- adapter diagnostics
- optional per-scale attention maps

## 主要風險

1. **Source pose leakage**
   - 避免直接 spatial injection 作為主線。
   - 使用 region tokens 與 visual cross-attn。

2. **Identity 不夠強**
   - 增加 visual tokens。
   - 加 FFN LoRA。
   - 擴大 blocks / scales。

3. **Prompt following 下降**
   - gate zero-init。
   - 只接 selected late blocks。
   - 不污染 text cross-attn。

4. **Face parsing mask noise**
   - 做 mask erosion / dilation ablation。
   - 小區域如眼睛在低 scale 可能太小，需設定 minimum area fallback。

5. **SAM3 feature 太偏 segmentation / objectness**
   - 需要比較 SAM3 global/region tokens 與 AdaFace feature 是否互補。
   - 可考慮 SAM3 + face recognition embedding dual-branch。

## v1 成功標準

- 同一 source ID 在不同 prompt / pose 下有可辨識相似度提升。
- 不明顯把 source pose 固定到 target generation。
- prompt 中表情、角度、背景仍可被遵循。
- 相比 T5 token replacement，不再出現 embedding OOD 導致的崩圖。
- checkpoint 掃描中存在中期最佳點，而不是訓練越久越像 source image template。
