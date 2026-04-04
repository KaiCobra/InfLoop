# ACM Multimedia — Conclusion Section Writing Guide

## Skill Purpose
When the user is writing or revising conclusion sections for ACM MM papers (especially training-free image editing / visual autoregressive models), apply these guidelines to produce reviewer-friendly, publication-quality writing.

---

## Patterns from 15+ Surveyed Papers (2023–2025)

### Papers Surveyed
FlowEdit, FireFlow, Stable Flow, KV-Edit, ReFlex, AREdit, LEDITS++, TurboEdit (ECCV & SIGGRAPH Asia), RF-Solver, Prompt-to-Prompt, MasaCtrl, VAR (NeurIPS 2024 Best Paper), Infinity (CVPR 2025 Oral), DiffEdit — spanning ACM MM, CVPR, ICCV, ICLR, NeurIPS, ICML, ECCV, SIGGRAPH.

---

## Key Finding: Length and Format

Conclusions in image editing papers are **short** — typically a single paragraph of 3–7 sentences (80–180 words).

- 1 paragraph (3–6 sentences): FlowEdit, FireFlow, KV-Edit, AREdit, LEDITS++, Infinity, RF-Solver
- 2 short paragraphs: Stable Flow (conclusion + numbered limitations), TurboEdit, VAR
- Separate limitations subsection: common for CVPR/ICCV, placed outside conclusion

---

## Universal 3-Part Structure

### Part A: Method Introduction (1–2 sentences)
State what you proposed and its defining characteristic.

**Opening phrases (ranked by frequency):**
- "We introduced [MethodName], a/an [key adjective] method for [task]."
- "In this paper, we present [MethodName], a [training-free/novel] [framework/approach] that [core mechanism]."
- "We propose [MethodName], which [verb] the challenge of [task] by [approach]."

**DO NOT** start with "In conclusion" or "To conclude" — considered amateurish.

### Part B: Technical Mechanism + Key Results (1–3 sentences)
Briefly describe *how* it works and cite the key empirical finding.

**Patterns:**
- "Our approach [mechanism description]. Evaluations on [benchmark] demonstrate [key result]."
- "By [gerund], our method achieves [what], while [maintaining/requiring no]."
- "Our key observation is that [insight]. This enables [capability]."

### Part C: Forward-Looking Statement (1 sentence)
Close with impact, inspiration, or a brief nod to future directions.

**Phrases (from real papers):**
- "We believe that [method/finding] will [verb] the development of..."
- "We hope our findings serve as a building block for future advancements in..."
- "Our work highlights the [untapped potential / viability] of..."
- "We envision that [mechanism] could inspire broader applications, such as [1–2 examples]."

---

## Handling Limitations

### Strategy 1: Separate Section (most common for CVPR/ICCV/ACM MM)
Place limitations in a dedicated subsection or appendix, not in the conclusion paragraph.
- KV-Edit: Limitations in Appendix D
- Stable Flow: Lists 3 numbered limitations after conclusion
- AREdit: Limitations in Analysis section

### Strategy 2: Integrated Single Sentence (elegant, for shorter papers)
A single sentence using the "strength-as-limitation" pivot:
- "While [MethodName]'s [strength] is beneficial for [common case], it can become a limitation when [edge case]."

### Strategy 3: No Explicit Limitations (acceptable for top papers)
- Infinity (CVPR 2025 Oral): Conclusion focuses entirely on achievement
- RF-Solver (ICML 2025): Ends with results claim

### Strategy 4: Limitations as Future Work (NeurIPS style)
Present limitations implicitly as promising future directions.

**Golden rule**: Never end on a limitation — always close on a positive, forward-looking note.

---

## Tone and Style Rules

### Confident but Not Overclaiming
- Use "achieves," "demonstrates," "enables" — **not** "solves" or "overcomes all challenges"
- "state-of-the-art performance" is acceptable when supported by experiments
- "matching or surpassing" preferred over "beating"
- Use "address" or "tackle," **not** "solve"

### Commonly Used Adjective Pairs
- "inversion-free, optimization-free and model agnostic"
- "efficient yet versatile and precise"
- "straightforward yet effective"
- "simple yet effective"
- "scalable and efficient"

### Forward-Looking Verbs
- "We believe / We hope / We envision / We anticipate"
- "may inspire / could inspire broader applications"
- "establishes a foundation for further advancements"

---

## Special Rules for Training-Free VAR Editing Papers

1. **Always restate "training-free"**: Every training-free paper explicitly restates this property in the conclusion.

2. **Explicitly contrast with diffusion-based editing**: Note that you bring this capability to autoregressive models — a less explored direction.

3. **Emphasize speed**: VAR models are inherently faster. Mention inference speed as a practical advantage.

4. **Frame the attention mechanism insight**: Highlight the *discovery* or *observation* that enables the method, not just the task.

5. **Acknowledge base model dependency honestly**: Editing quality is bounded by the base model's capabilities (e.g., Infinity's generation quality).

6. **Suggest one concrete broader application**: Video editing, multi-concept editing, etc. — NOT a laundry list.

7. **Position as opening a new direction**: VAR editing is less explored than diffusion editing; frame it as a new paradigm.

---

## DO's and DON'Ts Checklist

### DO:
1. Keep it to 1 paragraph (3–6 sentences) for the main conclusion
2. Open with "We introduce/present/propose [MethodName]..."
3. Restate the "training-free" and "inversion-free" properties
4. Mention the key technical insight (not just the task)
5. Include one quantitative or comparative claim
6. End with a forward-looking sentence
7. Use paired adjectives ("efficient yet precise")
8. Separate limitations into their own subsection if extensive
9. Frame limitations as specific failure cases, not general inadequacies

### DON'T:
1. Start with "In conclusion" or "To conclude"
2. Restate the abstract verbatim
3. Introduce new experimental evidence
4. Write more than 2 paragraphs
5. Use vague future work ("we plan to explore more")
6. Be overly apologetic about limitations
7. Include methodology details — conclusion is for impact, not process
8. Add broader impact/ethics within the conclusion (put it separately)
9. End on a limitation
10. Include references/citations in the conclusion (rare in surveyed papers)

---

## Template for ACM MM 2026 Training-Free VAR Editing Paper

```
\section{Conclusion}
\label{section:conclusion}

[Sentence 1: Method introduction — "We present [Name], a training-free [framework] for [task] based on [paradigm]."]
[Sentence 2: Key insight — "Our key observation is that [insight about attention/tokens]."]
[Sentence 3: Mechanism — "By [mechanism], our method [achieves what]."]
[Sentence 4: Results — "Experiments on [benchmark] demonstrate [SOTA claim], while requiring [no training/no inversion/minimal compute]."]
[Sentence 5: Forward-looking — "We believe [method] demonstrates the viability of [direction] and hope our findings inspire [broader application]."]
```

---

## Common Transition Phrases (Copy-Paste Ready)

**Opening:**
- "In this paper, we present..."
- "We introduced..."
- "In this work, we address the challenge of..."

**Mechanism:**
- "Our approach constructs..."
- "By [gerund], our method..."
- "Our key observation is that..."

**Results:**
- "Extensive experiments demonstrate that..."
- "Evaluations on [benchmark] show that..."
- "Experimental results confirm [SOTA] performance..."

**Limitations pivot:**
- "While [strength] is beneficial for [case], it can become a limitation when..."
- "We note that [specific case] remains challenging..."

**Closing:**
- "We believe [method] will [promote/advance] the development of..."
- "We hope our findings serve as a building block for..."
- "Our work highlights the viability of..."
