# P2P Pipeline - Quick Start Guide

## Files Created

```
InfLoop/
├── infinity/
│   ├── models/
│   │   ├── infinity_p2p.py          # New: P2P-enabled Infinity model
│   │   └── infinityFloat.py         # Base: indices-level rollback merge
│   └── utils/
│       └── bitwise_token_storage.py # New: Token storage class
├── tools/
│   └── run_p2p.py                   # New: P2P pipeline CLI
├── scripts/
│   └── infer_p2p.sh                 # New: P2P inference script
└── docs/
    └── P2P_README.md                # New: Full documentation
```

## 30-Second Start

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/infer_p2p.sh
```

This will:
1. Generate source image ("A cat is chasing a rat.")
2. Extract and save tokens
3. Generate target image ("A cat is chasing a butterfly.")
4. Save both images to `./outputs/p2p/`

## What Each File Does

### 1. BitwiseTokenStorage (`infinity/utils/bitwise_token_storage.py`)
- **Purpose:** Store and retrieve bitwise tokens from VAE quantizer
- **Key Methods:**
  - `save_tokens(scale_idx, tokens, mask)` - Store extracted tokens
  - `load_tokens(scale_idx, device)` - Load tokens for application
  - `save_to_file(path)` - Serialize to disk
  - `load_from_file(path)` - Deserialize from disk

### 2. infinity_p2p.py (`infinity/models/infinity_p2p.py`)
- **Purpose:** Infinity transformer with P2P token management
- **New Parameters in `autoregressive_infer_cfg()`:**
  - `p2p_token_storage` - BitwiseTokenStorage instance
  - `p2p_token_replace_prob` - Replacement probability (0-1)
  - `p2p_use_mask` - Enable mask-guided replacement
- **Token Flow:**
  1. During source generation: Save codes → BitwiseTokenStorage
  2. During target generation: Load codes → Replace with probability → Generate

### 3. run_p2p.py (`tools/run_p2p.py`)
- **Purpose:** CLI orchestrator for P2P workflow
- **Workflow:**
  - Parse arguments
  - Load models (tokenizer, VAE, Infinity)
  - **Step 1:** Generate source image + extract tokens
  - **Step 2:** Generate target image + apply tokens
  - Save results

### 4. infer_p2p.sh (`scripts/infer_p2p.sh`)
- **Purpose:** Easy-to-modify bash wrapper
- **Key Parameters:**
  ```bash
  source_prompt="A cat is chasing a rat."
  target_prompt="A cat is chasing a butterfly."
  num_source_scales=5
  p2p_token_replace_prob=0.5
  p2p_token_file="./tokens_p2p.pkl"
  ```

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────┐
│            P2P Image Editing Pipeline                  │
└─────────────────────────────────────────────────────────┘

┌─ STEP 1: SOURCE GENERATION ─┐
│                             │
│  Source Prompt              │
│       ↓                     │
│  Infinity Model             │
│  + scale 0→5:               │
│    - Generate indices       │
│    - Convert to codes       │
│    - Save to Storage ◄──┐   │
│  + scale 6→30:          │   │
│    - Generate normally      │
│       ↓                     │
│  Source Image +         BitwiseTokenStorage
│  Extracted Tokens       ├─ scale 0: [B,1,h,w,d]
│                         ├─ scale 1: [B,1,h,w,d]
│                         ├─ scale 2: [B,1,h,w,d]
│                         ├─ scale 3: [B,1,h,w,d]
│                         └─ scale 4: [B,1,h,w,d]
│                             (optional masks)
│                             
└─────────────────────────────┘

┌─ STEP 2: TARGET GENERATION ─┐
│                             │
│  Target Prompt              │
│       ↓                     │
│  Infinity Model             │
│  + scale 0→5:               │
│    - Generate indices       │
│    - Load source tokens ─┐  │
│    - Random replace mask │  │
│    - Apply replacement  ◄┤  │
│    - Convert to codes      │  │
│  + scale 6→30:          │  │
│    - Generate normally      │
│       ↓                     │
│  Target Image           BitwiseTokenStorage
│  (w/ source structure)  (reloaded)
│                             
└─────────────────────────────┘

Output: source.jpg, target.jpg, tokens_p2p.pkl
```

## Key Concepts

### Bitwise Tokens
- **What:** Float values ±0.1768 from VAE binary quantization
- **Shape:** `[B, 1, h, w, d]` per scale
- **Purpose:** Preserve structure while allowing semantic changes

### Probabilistic Replacement
- **How:** `mask = rand() < p2p_token_replace_prob`
- **Effect:** 
  - High prob (0.7-0.9) → Strong structure preservation
  - Low prob (0.1-0.3) → More target-specific content
  - Value 0.5 → Balanced mix

### Mask-Guided Editing (Extensible Interface)
- **Future Feature:** Selective replacement by spatial region
- **Current Support:** Infrastructure ready, awaiting mask computation
- **Example Use:** Only replace tokens in object regions (via segmentation)

## Common Commands

### Basic P2P Editing
```bash
python3 tools/run_p2p.py \
  --source_prompt "A dog on a bench" \
  --target_prompt "A cat on a bench" \
  --pn 1M \
  --model_path weights/infinity_2b_reg.pth \
  --vae_path weights/infinity_vae_d32reg.pth \
  --vae_type 32 \
  --text_encoder_ckpt weights/models--google--flan-t5-xl/snapshots/... \
  --text_channels 2048
```

### High-Fidelity Structure Preservation
```bash
python3 tools/run_p2p.py \
  ... \
  --num_source_scales 5 \
  --p2p_token_replace_prob 0.8  # High replacement
```

### Weak Structure Preservation (More Creative)
```bash
python3 tools/run_p2p.py \
  ... \
  --p2p_token_replace_prob 0.2  # Low replacement
```

### Reuse Tokens
```bash
# Run once to generate tokens
python3 tools/run_p2p.py \
  --source_prompt "A red car" \
  --target_prompt "A red car" \
  --p2p_token_replace_prob 0.0 \
  --p2p_token_file ./my_tokens.pkl

# Reuse same tokens with different target
python3 tools/run_p2p.py \
  --source_prompt "A red car" \
  --target_prompt "A blue car" \
  --p2p_token_file ./my_tokens.pkl \
  --p2p_token_replace_prob 0.5
```

## File Compatibility

| Component | Based On | Changes |
|-----------|----------|---------|
| `infinity_p2p.py` | `infinityFloat.py` | + P2P token save/load logic |
| `run_p2p.py` | `run_loop.py` | + Two-phase generation (source+target) |
| `infer_p2p.sh` | `infer.sh` | + P2P-specific parameters |
| `BitwisetokenStorage` | NEW | Token persistence layer |

## Validation

All files created are:
- ✅ Syntactically correct (py_compile check)
- ✅ Importable (module verification)
- ✅ Backward compatible (no breaking changes to existing code)
- ✅ Well-documented (docstrings and comments)

## Next Steps

1. **Test:** Run `bash infer_p2p.sh` to verify basic functionality
2. **Customize:** Edit prompts in `scripts/infer_p2p.sh`
3. **Experiment:** Adjust `--p2p_token_replace_prob` for different effects
4. **Extend:** Add mask computation for mask-guided editing

## Architecture Notes

The P2P pipeline is built on **indices-level token management**:

- **Why indices?** Integer operations (0/1) avoid quantization artifacts
- **When saved?** After transformer generates indices, before VAE conversion
- **When applied?** Before VAE conversion to codes
- **Upsampling?** Handled via `F.interpolate(mode='nearest')`

This design allows clean spatial manipulation without floating-point rounding errors.

## Support for Future Masks

The `BitwiseTokenStorage` class already supports mask storage:

```python
# Save with mask
storage.save_tokens(si, tokens, mask=segmentation_mask)

# Check mask availability
if storage.has_mask_for_scale(si):
    mask = storage.load_mask(si, device)
```

To implement mask-guided editing:
1. Compute mask (e.g., via segmentation model) on source image
2. Save with tokens: `storage.save_tokens(si, tokens, mask)`
3. Enable flag: `--p2p_use_mask 1`
4. Pipeline automatically applies: `mask = mask * random_mask`

## Troubleshooting

See [P2P_README.md](P2P_README.md) for detailed troubleshooting guide.
