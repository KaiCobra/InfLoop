# ACM Multimedia — Supplementary Material Writing Guide

## Skill Purpose
When the user is writing or revising supplementary material (appendix) for ACM MM papers — especially training-free image editing and visual generation — apply these guidelines to produce thorough, reviewer-friendly supplementary content.

---

## ACM MM 2024–2026 Supplementary Material Trends

### Observed Patterns from Top Papers

Papers surveyed: Prompt-to-Prompt (ICLR 2023), PnP Diffusion (ICCV 2023), MasaCtrl (ICCV 2023), Direct Inversion/PIE-Bench (ICLR 2024), InfEdit (CVPR 2024), LEDits++ (CVPR 2024), FlowEdit (ECCV 2024), TurboEdit (SIGGRAPH Asia 2024), ReFlex (NeurIPS 2024), AREdit (2024), KV-Edit (2025), RF-Inversion (2024), VAR (NeurIPS 2024 Best Paper), Infinity (CVPR 2025 Oral).

---

## Supplementary Material Anatomy — The 7 Standard Sections

Top venues consistently include these sections in descending order of importance. A strong supplementary for image editing has **4–7** of these.

### ① Algorithm Pseudocode (ESSENTIAL)
**Purpose**: Provide the complete, unambiguous procedure the main text couldn't fit.

**Rules**:
- Every algorithm referenced as "detailed in supplementary" MUST appear
- Use `\begin{algorithm}` with clean `\algorithmiccomment` annotations
- Cross-reference main text equations/sections: "See Eq.~3 in the main text"
- Include hyperparameter default values in comments
- Avoid mixing notation: if main text uses $k$ for scale index, supplementary must too

**Format**: 1–2 algorithms, each ≤ 25 lines. Longer algorithms lose reviewers.

**Best practice (FlowEdit, MasaCtrl)**: Place a sentence before the algorithm explaining what it does, then the algorithm block, then a 2–3 sentence interpretation.

### ② Additional Qualitative Results (HIGH IMPACT)
**Purpose**: Demonstrate generalization across diverse editing types and edge cases.

**Rules**:
- Show **all** editing categories from the benchmark (10 for PIE-Bench)
- Use `figure*` for full-width grids: 4–6 columns (methods) × 4–8 rows (examples)
- Include the source image, ground truth (if any), and 3+ baselines alongside your method
- **Caption must be descriptive**: "Row 1–2: object replacement; Row 3–4: style transfer. Red boxes highlight..."
- Never cherry-pick only successes — include at least 1–2 challenging cases where your method is imperfect
- Minimum 2 full-page figures for a competitive submission

**Typical layout (from Prompt-to-Prompt, PnP, LEDits++)**:
```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figs/supp_qualitative_1.pdf}
  \caption{Additional qualitative results on PIE-Bench. 
    Rows 1--3: object replacement. Rows 4--5: attribute modification.
    Our method preserves fine background details (e.g., the tree branches 
    in Row~2) while producing precise edits.}
  \label{fig:supp_qualitative}
\end{figure*}
```

### ③ Per-Category Quantitative Breakdown (HIGH IMPACT)
**Purpose**: Show robustness across diverse editing types, not just overall averages.

**Rules**:
- Break the main comparison table into **per-category** rows or columns
- PIE-Bench has 10 categories: report all 10 × N metrics (N = 4–6)
- Highlight which categories your method excels at and where it struggles
- Use `\footnotesize` or `\scriptsize` for large tables
- Bold best, underline second-best (consistent with main text convention)

**Format**: One large landscape table or 2–3 column-width tables grouped by metric type.

**Best practice (PIE-Bench paper, AREdit)**: Include a radar chart or heatmap alongside the table for visual summary. Not required, but impressive.

### ④ Implementation Details & Hyperparameters (STANDARD)
**Purpose**: Enable reproducibility beyond what the main text can contain.

**What to include**:
- Complete hyperparameter table (all defaults, search ranges tried)
- Model architecture details not in main text (layer counts, attention heads, etc.)
- Data preprocessing specifics (resolution, normalization, tokenization)
- Hardware specifics (GPU model, VRAM usage, inference time breakdown per phase)
- Software versions (PyTorch, CUDA, flash-attn)
- Random seed policy

**Format**: 1 paragraph + 1 table, or structured list with headings.

**Golden rule**: A reviewer should be able to reproduce your method from main text + supplementary alone.

### ⑤ Ablation Extensions (MEDIUM IMPACT)
**Purpose**: Supplement main-text ablations with deeper analysis.

**Typical content**:
- *Parameter sensitivity*: Sweep of key hyperparameters (e.g., threshold values, number of shared scales) with line plots
- *Component interaction analysis*: What happens when you combine/remove multiple components simultaneously
- *Failure mode drill-down*: Specific ablation on categories where performance drops
- *Attention map visualization*: Per-scale, per-block attention maps showing the masking mechanism in action

**Rules**:
- Each ablation extension should be self-contained with a brief motivation sentence
- Include figures AND numbers — figures alone lack rigor, numbers alone lack intuition
- Reference the corresponding main-text ablation: "Extending the analysis in Table~3..."

### ⑥ Attention / Intermediate Visualization (MEDIUM IMPACT)
**Purpose**: Provide interpretability and validate internal mechanisms.

**For image editing papers, show**:
- Attention maps at multiple scales (coarse → fine) for 3–4 examples
- Mask construction process: raw attention → IQR filtered → binarized
- Side-by-side: source attention vs. target attention for the same scale
- Heatmap overlays on the original image (use `jet` or `viridis` colormap)

**Format**: Multi-row figure with scale progression, or grid comparing mask variants.

**Best practice (Prompt-to-Prompt)**: Annotate which words each attention map corresponds to.

### ⑦ Additional Application Modes (OPTIONAL but differentiating)
**Purpose**: Show versatility of the framework beyond the main experimental setting.

**Typical content for editing papers**:
- Text-to-Image (T2I) editing mode (generation-only, no real image input)
- Multi-object editing examples
- Sequential/iterative editing chains
- Cross-domain editing (photographic → artistic style)
- High-resolution or different-resolution results

**Rules**:
- Don't over-claim: clearly state these are preliminary demonstrations
- Each mode should have at least 3 diverse examples
- If the main text promises this section, it MUST appear

---

## Structure Template for ACM MM Supplementary

```latex
\appendix

\section{Algorithm Details}
\label{sec:supp_algorithms}

\subsection{[Algorithm 1 Name]}
\label{sec:supp_alg1}
[1–2 sentences context] 
\input{equations/algor1}
[2–3 sentences interpretation]

\subsection{[Algorithm 2 Name]}
\label{sec:supp_alg2}
\input{equations/algor2}

\section{Additional Experimental Results}
\label{sec:supp_results}

\subsection{Per-Category Quantitative Analysis}
\label{sec:supp_per_category}
[Table with 10-category breakdown]
[2–3 paragraph analysis: where method excels, where it struggles, why]

\subsection{Additional Qualitative Comparisons}
\label{sec:supp_qualitative}
[Full-width figure grids, 2+ pages]
[Per-figure caption with category labels and key observations]

\subsection{Failure Case Analysis}
\label{sec:supp_failure}
[3–5 failure examples with analysis of WHY they fail]
[Connect to limitations in main text conclusion]

\section{Extended Ablation Studies}
\label{sec:supp_ablation}

\subsection{Parameter Sensitivity Analysis}
\label{sec:supp_sensitivity}
[Sweep plots for key hyperparameters]

\subsection{Scale-Level Analysis}
\label{sec:supp_scale}
[Visual analysis of per-scale behavior]

\subsection{Attention Mask Visualization}
\label{sec:supp_attention_vis}
[Multi-scale attention maps, mask construction pipeline]

\section{Additional Application Modes}
\label{sec:supp_applications}

\subsection{Text-to-Image Editing}
\label{sec:supp_t2i}
[T2I editing demonstration if promised in main text]

\section{Implementation Details}
\label{sec:supp_impl}
[Complete hyperparameter table, hardware, software versions]
```

---

## Writing Rules for Supplementary Material

### Cross-Referencing
- **Always** cross-reference the main text: "As described in Sec.~4.2 of the main text..."
- **Always** explain WHY something is in supplementary: "Due to space constraints, we provide..."
- Use consistent notation — if main text uses $\tau_k$, supplementary must use $\tau_k$
- Reference supplementary sections from within supplementary (internal links)

### Tone and Depth
- Supplementary can be more detailed and technical than the main text
- Explain "why" not just "what" — reviewers use supplementary to verify claims
- Don't repeat main text verbatim; add new information or deeper analysis
- Treat supplementary as "proof appendix" — back up every claim in main text

### Figure Quality
- **Same quality as main text figures** — LaTeX-rendered, vector graphics preferred
- Clear axes labels, readable font sizes (even at 50% zoom)
- Consistent colormaps and color conventions across all figures
- Include figure numbers that continue from the main text (LaTeX handles this with `\appendix`)

### Table Format
- Continue numbering from main text
- Use same metric conventions (↑/↓ arrows, bold/underline)
- `\small` or `\footnotesize` for large tables is acceptable in supplementary
- Keep column alignment consistent with main text tables

### Page Budget
- ACM MM sigconf: no strict supplementary page limit (unlike main text 6–8 pages)
- **Target 3–6 pages** for a well-organized supplementary
- More than 8 pages supplementary suggests content that should be in main text
- Reviewers spend 5–15 minutes on supplementary — front-load the most important content

---

## Common Reviewer Concerns Addressed by Supplementary

| Reviewer Concern | Supplementary Section to Add |
|---|---|
| "Are results cherry-picked?" | § Additional Qualitative (diverse examples, include imperfect ones) |
| "How robust across edit types?" | § Per-Category Breakdown (all 10 PIE-Bench categories) |
| "Can I reproduce this?" | § Implementation Details (hyperparameters, hardware, seeds) |
| "What about failure cases?" | § Failure Analysis (with root cause discussion) |
| "Is the threshold sensitive?" | § Parameter Sensitivity (sweep plots) |
| "How does the mask look?" | § Attention Visualization (multi-scale maps) |
| "Does it work for T2I?" | § Additional Applications (if promised in main text) |
| "What do intermediate steps look like?" | § Attention/Mask Construction Pipeline visualization |

---

## Anti-Patterns to Avoid

1. **Empty promises**: If main text says "see supplementary for X," X MUST be there. Broken references are a red flag.
2. **Dump of figures with no analysis**: Every figure needs a substantive caption + 1–2 analysis sentences.
3. **Repeating main text**: Don't copy-paste methodology or results tables. Add NEW content.
4. **Inconsistent notation**: Different symbols for the same variable between main/supp is a common error.
5. **Low-quality figures**: Blurry screenshots, inconsistent spacing, missing labels. Supplementary figures matter.
6. **No failure cases**: Omitting failures signals cherry-picking to reviewers.
7. **Missing ablation promised in main text**: If main text says "we ablate X in supplementary," it must be there.
8. **Supplementary longer than main text**: Suggests poor main text organization. Keep focused.

---

## Prioritization for Camera-Ready vs. Initial Submission

### Must-have for initial submission:
- ✅ All algorithms referenced by main text
- ✅ Per-category quantitative breakdown
- ✅ Additional qualitative results (2+ figures)
- ✅ All sections promised in main text

### Nice-to-have (add during revision / camera-ready):
- Additional baselines not in main comparison
- Larger qualitative figures with more examples
- Parameter sensitivity sweeps
- User study details (protocol, questionnaire)
- T2I mode or other application extensions

---

## ACM-Specific Formatting Notes

- Supplementary goes in `\appendix` block, sections become "A", "B", "C" etc.
- Use `\section{...}` normally — LaTeX converts to appendix letters automatically
- Figures/tables continue numbering from main text (no prefix needed)
- Keep the same `\documentclass[sigconf]{acmart}` — don't switch document class
- `\label{sec:supp_*}` convention keeps supplementary labels distinct from main text
- If uploading separately: match the main text template exactly (fonts, margins)
