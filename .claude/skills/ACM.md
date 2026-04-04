# ACM Multimedia Methodology Writing Guide

## Skill Purpose
When the user is writing or revising methodology sections for ACM MM papers (especially training-free image editing / visual autoregressive models), apply these guidelines to produce reviewer-friendly, publication-quality writing.

---

## Paper Title
**Understanding Before Editing: Unleashing VAR for Training-Free Text-Guided Image Editing with Cross-Modal Text-Visual Attention**

---

## ACM MM 2024-2026 Trends & Reviewer Expectations

### Conference Stats & Key Facts
- **ACM MM 2024** (Melbourne): 25% acceptance (2,145/8,556). Image editing highlight: VICTORIA (Attentive Linguistic Tracking for Multi-Object Editing)
- **ACM MM 2025** (Dublin): 1,250 regular papers. VAR-related: "Safe-BVAR" (watermarking for bitwise VAR). Topic: "Multimedia Generative and Foundation Models" is first-class
- **ACM MM 2026**: OpenReview, double-blind, **6-8 pages + up to 2 pages for references only**. Explicitly welcomes "bold, forward-looking ideas supported by early but promising results, particularly in areas involving multimodal AI"
- **Context**: VAR won NeurIPS 2024 Best Paper; Infinity is CVPR 2025 Oral — VAR-based editing is timely and novel

### Official ACM MM Reviewer Evaluation Criteria
1. **Novelty**: What does the paper contribute? Is it valuable for the multimedia research community?
2. **Scientific Rigor**: Are experiments well-designed, sufficient, and reproducible? Released code/data?
3. **Scope & Topical Fit**: Does the paper involve multi-modal data? Papers must relate to multimedia
4. **Presentation Quality**: Writing clarity, logical flow, adequate contextualization vs. prior work
- Key standard: "Central claims are adequately supported with evidence."
- **CRITICAL — Multimodal scope**: ACM MM reviewers specifically check topical fit. Frame contributions in terms of **cross-modal understanding** (text-image attention interaction), not purely visual processing

### What Reviewers Look For in Methodology
1. **Clear problem formulation** — state what fails in prior work and WHY your approach avoids it
2. **Mathematical precision** — every design choice backed by notation, not just prose
3. **Pipeline figure** — essential; the overview figure is the first thing reviewers study
4. **Algorithm boxes** — at least one pseudocode Algorithm environment
5. **Component justification** — each module should have a clear "why" (not just "what")
6. **Ablation-ready structure** — present components so each can be independently evaluated
7. **Comparison woven in** — subtly contrast with baselines within the method section
8. **Notation consistency** — use the same symbols throughout; define once, use everywhere

### Common Pitfalls to Avoid
1. Jumping straight into equations without motivation (always explain WHY before HOW)
2. Missing or inadequate Preliminaries section (must formally define VAR, BSQ-VAE, attention)
3. Unclear distinction between existing vs. novel components (use "we propose", "our key insight is")
4. Overloading implementation details before the conceptual framework is clear
5. Ignoring the multimedia/multimodal angle (frame as cross-modal text-image understanding)
6. Vague descriptions of attention manipulation (specify: which layers, which scales, what threshold, how masks are derived)
7. No failure case discussion — acknowledge limitations explicitly

### Writing Style Rules
- **Lead with motivation, then mechanism**: "To preserve background fidelity without inversion, we..." not "We use X which does Y"
- **One paragraph = one idea**: Don't pack multiple mechanisms into one paragraph
- **Equation → prose interpretation**: Every equation should be followed by an intuitive explanation
- **Active voice**: "We extract..." not "The extraction is performed..."
- **Quantify claims**: "reduces inference steps from 28 to 13" not "significantly reduces computation"
- **Avoid vague superlatives**: "achieves competitive/strong/state-of-the-art" with numbers, not "significantly better"
- **Emphasize cross-modal**: Frame attention mechanisms as "cross-modal text-visual interaction" for ACM MM scope

---

## Project-Specific Methodology Components

### This Paper's Core Story

**Narrative arc for reviewers:**
1. Diffusion editing needs inversion → inversion has cumulative errors
2. VAR generates discrete tokens scale-by-scale → no trajectory to invert
3. Cross-attention in VAR reveals text-token spatial correspondence → use it for masks
4. "Understanding before editing": extract attention to understand spatial semantics, THEN selectively edit tokens
5. Vertical pipeline: process all three passes (source/guided/target) per-scale, not horizontally

### Key Technical Contributions to Formalize

#### A. IQR-Filtered Attention Extraction (Dynamic Mask Thresholding)
- **Problem**: Infinity's raw attention maps exhibit a global layer effect (anchor bias) — some transformer blocks produce attention distributions dominated by global bias rather than semantic content. Directly averaging all blocks yields noisy, unreliable spatial masks.
- **Solution**: IQR (Interquartile Range) filtering to robustly remove outlier blocks before aggregation
- **Why IQR**: Non-parametric, robust to skewed distributions, does not assume Gaussian — appropriate for heterogeneous attention patterns across transformer blocks
- **Algorithm** (must be presented as Algorithm 1 in paper):
  ```
  Input: attention maps {A_b}_{b=1}^{B} from B blocks, focus token indices, percentile p
  1. For each block b, extract spatial attention: a_b = mean(A_b[:, focus_tokens]) → (H, W)
  2. Compute reference: a_mean = mean({a_b})
  3. For each block b: MSE_b = mean((a_b - a_mean)^2)
  4. Compute Q1, Q3, IQR = Q3 - Q1 of {MSE_b}
  5. Keep blocks where MSE_b ≤ Q3 + 1.5 × IQR
  6. a_filtered = mean(kept blocks)
  7. Focus mask: M_focus = (a_filtered ≥ percentile(a_filtered, p))
  8. Preserve mask: M_preserve = (a_filtered ≤ percentile(a_filtered, 100-p))
  Output: M_focus, M_preserve
  ```
- **Key property**: single threshold p controls both masks symmetrically; no per-category tuning needed

#### B. Cross-Modal Attention Masking
- Source focus mask: top-(100-p)% attention to source focus words → editable region
- Low-attention preserve mask: bottom-(100-p)% attention → background anchor
- Target focus mask: top-(100-p)% attention to target focus words
- Union mask: `M_union = M_src_focus ∪ M_tgt_focus`; replacement mask = `~M_union`

#### C. Vertical Scale-Wise Pipeline (per-scale processing)
- **Key design change**: Instead of running all scales for source gen, then all scales for guided gen, then all scales for target gen (horizontal), process **vertically**: for each scale k, execute source gen → guided gen → target gen, then proceed to scale k+1
- **Why vertical**:
  - Attention masks at scale k can immediately inform token replacement at scale k
  - Reduces peak memory: only one scale's attention maps need to reside in memory at a time
  - More natural correspondence: each scale's source attention directly guides the same scale's target generation
  - Aligns with VAR's inherent coarse-to-fine structure — decisions at coarse scales propagate naturally
- **Implementation**: For each scale k:
  1. Source gen at scale k → extract source cross-attention → compute source focus mask
  2. Guided gen at scale k (with preserve mask anchoring bg tokens) → extract target cross-attention → compute target focus mask
  3. Compute union mask → apply selective token replacement for target gen at scale k
  4. Move to scale k+1

#### D. Source Image Encoding (discrete injection)
- Source image is encoded via BSQ-VAE into per-scale discrete bitwise tokens: `{b^{(k)}}_{k=1}^{S} = Q_bsq(E_vae(I_src))`
- These tokens are used for direct token replacement (pixel-faithful, prompt-independent)
- **Note on continuous injection**: Injecting continuous VAE encoder features (before quantization) as `summed_codes` mixing causes visible image distortion — the soft blending between source features and generation dynamics introduces artifacts. We therefore use only discrete token replacement for structural preservation. (Ablation figure will demonstrate this distortion.)

### Notation Conventions (maintain consistency)
- `I_src`: source image
- `P_s, P_t`: source/target prompts
- `f_s`: source focus words; `f_t`: target focus words
- `b^{(k)}`: bitwise tokens at scale k (from source image BSQ-VAE encoding)
- `M^{(k)}_{focus}`: focus mask at scale k
- `M^{(k)}_{preserve}`: preserve mask at scale k
- `M^{(k)}_{replace}`: final replacement mask at scale k
- `A^{(k)}_{cross}`: cross-attention map at scale k
- `S`: total number of scales; `N`: number of full-replace scales
- `B`: number of transformer blocks; `p`: percentile threshold

---

## Figures to Include in Methodology

1. **Pipeline overview figure** (essential, full-width)
   - **Vertical layout**: show per-scale processing (scale 1 → scale 2 → ... → scale S)
   - At each scale: source gen → guided gen → target gen (three columns or three arrows)
   - Show attention mask extraction and application within each scale
   - Show source image tokens feeding into replacement at each scale

2. **IQR filtering visualization** (recommended)
   - Show raw per-block attention maps (some with global bias / anchor effect)
   - Show MSE distribution with IQR cutoff line
   - Show filtered mean vs. unfiltered mean attention map

3. **Attention mask visualization** (recommended)
   - Source focus mask, target focus mask, union mask, replacement mask
   - Show at multiple scales (coarse → fine)

4. **Continuous vs. discrete injection comparison** (ablation figure)
   - Show distortion from continuous injection vs. clean result from discrete-only

---

## Common Reviewer Concerns to Preemptively Address

1. **"How is this different from AREdit?"** → We use cross-modal attention maps for explicit spatial masking at each scale; AREdit caches token statistics and requires per-category tuning. **Note: AREdit has not released source code, so direct reproduction-based comparison is not possible; we compare based on their reported numbers and methodology.**
2. **"Why not just use a segmentation model for masks?"** → Our masks are derived from the model's own cross-attention, requiring no external modules
3. **"What about computational cost?"** → VAR generates in ~13 scales (vs 28-50 denoising steps in diffusion); ~10.0s on single GPU with 2B model
4. **"How does this handle diverse edit types?"** → Single-focus fallback mechanism handles add/remove/change uniformly
5. **"Is the percentile threshold sensitive?"** → Show ablation on threshold values; the symmetric design makes it robust
6. **"Why IQR filtering instead of learned block weighting?"** → Training-free constraint; IQR is non-parametric and requires no data-driven tuning

---

## Detailed Technical Pipeline

### Vertical Per-Scale Processing

```
For each scale k = 1..S:
  ┌─ Source Pass ─────────────────────────────────────┐
  │  Generate source tokens at scale k                │
  │  Extract source cross-attention A^{(k)}_{src}     │
  │  If k ≥ N: compute M^{(k)}_{src_focus},           │
  │            M^{(k)}_{preserve} via IQR filtering    │
  └───────────────────────────────────────────────────┘
           ↓ (preserve mask)
  ┌─ Guided Pass ────────────────────────────────────┐
  │  Generate with target prompt at scale k           │
  │  Background tokens (preserve mask=True) anchored  │
  │    to source image tokens b^{(k)}                 │
  │  Extract target cross-attention A^{(k)}_{tgt}     │
  │  If k ≥ N: compute M^{(k)}_{tgt_focus}            │
  └───────────────────────────────────────────────────┘
           ↓ (union mask)
  ┌─ Target Pass ────────────────────────────────────┐
  │  M^{(k)}_{union} = M^{(k)}_{src_focus} ∪          │
  │                     M^{(k)}_{tgt_focus}            │
  │  M^{(k)}_{replace} = ¬M^{(k)}_{union}             │
  │  If k < N: idx = b^{(k)}  (100% source tokens)    │
  │  If k ≥ N: idx = M^{(k)}_{replace} ? b^{(k)}      │
  │                   : gen^{(k)}_{target}              │
  └───────────────────────────────────────────────────┘
           ↓
  Proceed to scale k+1
```

### IQR Attention Filtering (robustness mechanism)
```python
# Stack attention maps from all blocks: [num_blocks, H, W]
# Compute per-block MSE from mean
# Keep blocks where MSE ≤ Q3 + 1.5 × IQR
# Return filtered mean → more stable spatial masks
# Motivation: Infinity's transformer blocks exhibit global layer effect
# (anchor bias) where some blocks' attention is dominated by positional
# patterns rather than semantic content. IQR removes these outliers.
```

### Computational Complexity
- 3 forward passes total (interleaved per-scale): source gen, guided gen, target gen
- Attention capture: O(S × B × H × W) per pass (S scales, B blocks)
- Mask computation: O(S × H × W) with IQR filtering
- Total inference: ~10.0s on single GPU (2B model, 13 scales)

---

## Suggested Methodology Subsection Outline

```latex
\section{Methodology}

\subsection{Pipeline Overview}
% Reference to pipeline figure (Fig.2)
% High-level: vertical per-scale processing
% Three interleaved passes per scale: source → guided → target
% Emphasize: inversion-free, tuning-free, "understanding before editing"

\subsection{Preliminaries: Visual Autoregressive Modeling}
% VAR architecture, scale-wise generation, BSQ-VAE tokenization
% Define: scale schedule, cross-attention formulation
% Key point: discrete tokens at each scale, no noise trajectory
% Source image encoding: BSQ-VAE → per-scale discrete tokens

\subsection{Dynamic Mask Thresholding with IQR Filtering}
% Problem: Infinity's attention maps have global layer effect (anchor bias)
% Why IQR: non-parametric, robust to skewed distributions, training-free
% Algorithm 1: IQR-filtered attention mask extraction (pseudocode)
% Symmetric threshold: single p controls both focus and preserve masks
% Visual: raw vs. filtered attention maps

\subsection{Cross-Modal Attention Masking}
% Extract cross-attention during source/target generation at each scale
% High-attention focus mask: identifies editable regions
% Low-attention preserve mask: identifies background for anchoring
% IQR filtering integrated into mask computation
% Union combination: M_replace = ¬(M_src_focus ∪ M_tgt_focus)
% Key: single threshold p controls both masks symmetrically

\subsection{Scale-Wise Selective Token Replacement}
% Vertical processing: source → guided → target at each scale
% Coarse scales (k < N): full source token replacement → structural anchoring
% Fine scales (k ≥ N): mask-guided selective replacement
% Algorithm 2: per-scale editing procedure (pseudocode)
% Background tokens → source image tokens (preservation)
% Focus tokens → free target generation (editing)

\subsection{Implementation Details}
% Hardware: NVIDIA RTX 5090
% Model: Infinity 2B with BSQ-VAE (d=32)
% Inference time, memory usage
% Detailed hyperparameters in supplementary
```

---

## Formally Compared Methods (in comparison table)

### Diffusion-based
- Prompt-to-Prompt (P2P) — Diffusion, 2B
- Pix2Pix-Zero — Diffusion, ~1B
- MasaCtrl — Diffusion, ~1B
- PnP — Diffusion, ~1B
- PnP Inversion — Diffusion, ~1B
- LEDits++ — Diffusion, ~1B

### Flow-based
- FlowEdit — Flow, 12B
- RF-Inversion — Flow, 12B
- ReFlex — Flow, 12B
- FireFlow — Flow, 12B

### Ours
- VAR (Infinity), 2B — S.D. 0.0338, PSNR 20.44, SSIM 0.7837, LPIPS 0.1305, CLIP whole 0.2566, CLIP edited 0.2269, Inf. Time 10.0s

**Note on AREdit**: AREdit (ICCV 2025) is discussed in Related Works as the closest VAR-based editing work, but has **not released source code**. Therefore, direct reproduction-based comparison is not feasible. When mentioning AREdit in the paper, reference their published methodology and reported results, but do not claim direct experimental comparison.

---

## ACM MM Formatting Reminders
- Page limit: 6-8 pages + up to 2 pages for references only (ACM MM 2026)
- Use `\begin{algorithm}` environment for pseudocode
- Figures: use `\begin{figure*}` for full-width pipeline overview
- Tables: comparison table should include inference time, model size, and both edit quality + preservation metrics
- CCS concepts and keywords must be accurate
- Anonymous submission: remove all author-identifying information
- Use `\cite{}` consistently; avoid "our previous work [X]" patterns

---

## Key References for Methodology Comparison

### ACM MM Image Editing Papers (2024-2025)
- **VICTORIA** (ACM MM 2024): Attentive linguistic tracking for training-free multi-object editing
- **Safe-BVAR** (ACM MM 2025): Watermarking for bitwise VAR — shows VAR is a recognized topic at ACM MM

### Top Methodology References
- **VAR** (NeurIPS 2024 Best Paper): Paradigm challenge → next-scale prediction → architecture → scaling laws
- **Infinity** (CVPR 2025 Oral): Bitwise token prediction + self-correction → high-res generation
- **Prompt-to-Prompt** (ICLR 2023): 3 clean control mechanisms, each with own subsection + math + visual example
- **AREdit** (ICCV 2025): Cache-based VAR editing — closest competitor; weakness = per-category tuning; **no source code released**

### Reviewer Guidelines Sources
- ACM MM 2025 Reviewer Guidelines: https://acmmm2025.org/reviewer-and-area-chair-guidelines/
- ACM MM 2026 CFP & Review Process: https://2026.acmmm.org/site/cfp-guidelines.html
- Autoregressive Models in Vision Survey (TMLR 2025): https://github.com/ChaofanTao/Autoregressive-Models-in-Vision-Survey
