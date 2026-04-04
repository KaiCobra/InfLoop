# ACM Multimedia — Experiment Section Writing Guide

## Skill Purpose
When the user is writing or revising experiment sections for ACM MM papers (especially training-free image editing / visual autoregressive models), apply these guidelines to produce reviewer-friendly, publication-quality writing.

---

## ACM MM 2024–2025 Experiment Section Trends

### Observed Patterns from Highlight & Best Papers

1. **Quantitative-first, visual-second**: Lead each comparison with a full metrics table, then provide qualitative figures. Reviewers scan numbers before studying images.

2. **Per-category breakdown**: Top papers (VICTORIA, AREdit, InstructPix2Pix) break PIE-Bench results into per-edit-type tables (10 categories), not just overall averages. This shows robustness across diverse editing scenarios.

3. **Metric grouping convention**: Present metrics in two logical groups:
   - **Structure preservation** (background fidelity): Structure Distance, PSNR, SSIM, LPIPS
   - **Edit quality** (semantic alignment): CLIP whole-image similarity, CLIP edited-region similarity
   - Some papers add FID or user study scores as a third group

4. **Ablation depth**: Highlight papers typically include 3–5 ablation experiments, each isolating one design choice. One-variable-at-a-time is the gold standard. Multi-variable ablations are acceptable only when components interact (with interaction analysis).

5. **Ablation presentation**: Use compact tables (2–4 rows per ablation) with a "Full model" baseline row. Bold the best result. Include a brief analysis paragraph after each table explaining *why* each component matters.

6. **Qualitative comparison figures**: Full-width `figure*` with 5–7 methods side-by-side on 3–4 diverse examples. Red boxes or zoom-ins highlight differences. Caption should point out specific failure modes of baselines.

7. **User study**: ACM MM 2024–2025 highlights increasingly include user studies (20–50 participants, pairwise preference or Likert scale). Even a small study strengthens the "multimedia" angle.

8. **Inference efficiency**: A dedicated row/column for inference time and model size. ACM MM reviewers care about practicality — 10s on a single GPU is a strong selling point vs. 12B flow models.

9. **Failure case analysis**: At least one paragraph (or supplementary figure) discussing failure modes. Reviewers penalize papers that appear to cherry-pick only successes.

10. **Reproducibility signals**: Mention seed, number of runs, standard deviation where applicable. State clearly which numbers are reproduced vs. cited from original papers.

---

## Recommended Experiment Section Structure

```latex
\section{Experiments and Discussions}

\subsection{Experimental Settings}
% 3 paragraphs: benchmark, metrics, implementation details
% Keep implementation details brief if already in methodology — just reference §4.6

\subsection{Comparison with State-of-the-Art Methods}
% 1. Full metrics table (Table 1) — already exists as relatedWorksComparison.tex
% 2. Analysis paragraph: highlight where our method excels and where it trades off
% 3. Full-width qualitative figure (Fig. X) — 5-7 methods, 3-4 examples
% 4. Brief discussion of model size advantage (2B vs 12B)

\subsection{Ablation Studies}
% 3-5 compact ablation tables, each with analysis paragraph
% Structure: "Full model" row + variants with one component removed/changed
% End with a summary paragraph tying ablations back to contributions

\subsection{Limitations and Discussion}
% Failure cases, editing types where method struggles
% Honest assessment — reviewers reward transparency
```

---

## Writing Rules for Each Subsection

### §5.1 Experimental Settings

**PIE-Bench paragraph:**
- Cite PIE-Bench properly, state 700 images / 10 edit types
- List the 10 categories briefly (or reference table)
- Mention resolution (512×512 or actual)

**Metrics paragraph:**
- Define each metric in one sentence with citation
- Group: {Structure Distance, PSNR↑, SSIM↑, LPIPS↓} for preservation; {CLIP whole↑, CLIP edited↑} for edit quality
- State which direction is better (↑/↓) — reviewers should never have to guess

**Implementation Details paragraph:**
- If already in §4 methodology, keep this to 2–3 sentences referencing that section
- Add only experiment-specific details: GPU, inference time, seed, batch size
- "All experiments use a single seed unless otherwise stated. We report metrics averaged over all 700 test cases."

### §5.2 Comparison with SOTA

**Table analysis — the "narrative" approach:**
- Don't just list numbers. Tell a story:
  1. Overall ranking: "Our method achieves the best/second-best Structure Distance among all methods"
  2. Group comparison: "Compared to diffusion-based methods (2B), we improve PSNR by X while matching CLIP scores"
  3. Scale comparison: "Despite using a 2B model, we achieve competitive results with 12B flow-based methods"
  4. Trade-off acknowledgment: "FlowEdit achieves higher CLIP edited score, but at 6× the model size and requiring ODE inversion"

**Qualitative figure rules:**
- Choose examples that span different edit types (object replacement, style change, addition, removal)
- Include at least one example where our method clearly wins AND one where baselines are competitive
- Use red boxes / zoom-ins on specific regions
- Caption must reference specific visual differences, not generic "our method produces better results"

### §5.3 Ablation Studies

**Format for each ablation:**
```latex
\noindent\textbf{Effect of [component name].}
[1-2 sentence motivation for why this component matters]

\begin{table}[t]
  \caption{Ablation on [component]. Best in \textbf{bold}.}
  % compact table: 2-4 rows, 4-6 metric columns
\end{table}

[2-3 sentence analysis: which metrics changed, why, what this tells us]
```

**Golden rules:**
- Each ablation must map to a claim in contributions or methodology
- "w/o IQR filtering" must exist if you claim IQR is important
- Show both preservation AND edit quality metrics — a component that improves one while hurting the other reveals a trade-off, which is informative
- Use consistent naming: "Full model", "w/o X", "X → Y variant"

### §5.4 Limitations

**What to include:**
- 1–2 failure case examples (figure in supplementary is fine, reference it)
- Specific editing types where performance drops
- Honest comparison: "On category X, flow-based methods outperform ours due to..."
- Future directions (1–2 sentences, no speculation)

---

## Ablation Study Design Principles (ACM MM Standard)

### One-Variable-at-a-Time Protocol
1. Establish a "Full model" configuration (your best/default settings)
2. For each ablation, change ONLY the target variable, keep everything else identical
3. Run on the SAME test set (all 700 PIE-Bench cases or a representative subset)
4. Report the SAME metrics across all ablations

### Statistical Rigor
- If variance is high, report mean ± std over 3 seeds
- For PIE-Bench (700 deterministic cases with fixed seed), single-run is acceptable if seed is fixed
- State clearly: "We use seed=1 for all experiments unless otherwise noted"

### Ablation Table Format
```
| Setting              | S.D.↓  | PSNR↑  | SSIM↑  | LPIPS↓ | CLIP-w↑ | CLIP-e↑ |
|---------------------|--------|--------|--------|--------|---------|---------|
| Full model          | 0.0338 | 20.44  | 0.7837 | 0.1305 | 0.2566  | 0.2269  |
| w/o component A     | ...    | ...    | ...    | ...    | ...     | ...     |
| variant B           | ...    | ...    | ...    | ...    | ...     | ...     |
```

### What Makes a Good Ablation (Reviewer Perspective)
- **Validates the contribution**: Each novel component should have a corresponding ablation
- **Shows non-trivial impact**: >1% relative change on at least one metric
- **Reveals trade-offs**: If removing a component improves one metric but hurts another, discuss why
- **Includes baselines**: "w/o our component" should resemble a simpler/naive approach that a reviewer might suggest

---

## Common Reviewer Questions to Preemptively Address

1. **"Why not compare with AREdit?"** → AREdit has not released code; we compare on reported metrics and discuss methodological differences
2. **"Are the improvements statistically significant?"** → On 700-sample PIE-Bench with fixed seed, report exact numbers; for user studies, report p-values
3. **"How sensitive is the method to hyperparameters?"** → Ablation on threshold percentile directly addresses this
4. **"What about editing types not in PIE-Bench?"** → Reference supplementary for additional qualitative results
5. **"Inference time comparison?"** → Include in main table; highlight single-GPU, no-inversion advantage
6. **"Why only 2B model?"** → Discuss scaling potential; note that 12B flow baselines have 6× parameters

---

## Formatting Reminders (ACM sigconf, two-column)

- Tables: use `\begin{table}[t]` for single-column, `\begin{table*}[t]` for full-width
- Figures: use `\begin{figure*}[t]` for qualitative comparison (needs full width for 5+ methods)
- Bold best results, underline second-best (or use colored cells)
- Table captions go ABOVE the table; figure captions go BELOW
- Use `\small` or `\footnotesize` in tables to fit within column width
- Metric arrows (↑/↓) in column headers, not in every cell
