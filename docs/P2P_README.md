# Prompt-to-Prompt (P2P) Image Editing Pipeline

## Overview

This pipeline enables **prompt-to-prompt image editing** by leveraging structure preservation from source images. The key idea is to extract bitwise tokens (±0.1768) from the source image generation process and probabilistically apply them to the target image generation.

### Architecture

The P2P pipeline consists of:

1. **BitwiseTokenStorage** (`infinity/utils/bitwise_token_storage.py`)
   - Manages extraction and storage of bitwise tokens from VAE quantizer
   - Supports optional mask storage for future mask-guided editing
   - Serializable to disk for token reuse

2. **infinity_p2p.py** (`infinity/models/infinity_p2p.py`)
   - Extended Infinity transformer with P2P token management
   - Based on infinityFloat.py (indices-level rollback merge)
   - Saves tokens during source image generation
   - Applies tokens during target image generation

3. **run_p2p.py** (`tools/run_p2p.py`)
   - CLI interface for P2P editing workflow
   - Orchestrates source and target image generation
   - Handles token file I/O

4. **infer_p2p.sh** (`scripts/infer_p2p.sh`)
   - Bash wrapper with default parameters
   - Easy-to-customize example for users

## Usage

### Basic P2P Editing

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/infer_p2p.sh
```

Or with custom prompts:

```bash
python3 tools/run_p2p.py \
  --source_prompt "A cat is chasing a rat." \
  --target_prompt "A cat is chasing a butterfly." \
  --pn 1M \
  --model_path weights/infinity_2b_reg.pth \
  --vae_path weights/infinity_vae_d32reg.pth \
  --vae_type 32 \
  --text_encoder_ckpt weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
  --text_channels 2048 \
  --cfg 4 \
  --tau 0.5 \
  --seed 1 \
  --save_file ./outputs/p2p/ \
  --num_source_scales 5 \
  --p2p_token_replace_prob 0.5 \
  --p2p_token_file ./tokens_p2p.pkl
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--source_prompt` | str | Required | Source prompt for initial image generation |
| `--target_prompt` | str | Required | Target prompt for edited image generation |
| `--num_source_scales` | int | 5 | Number of scales to extract tokens from source |
| `--p2p_token_replace_prob` | float | 0.5 | Probability of replacing target token with source token |
| `--p2p_token_file` | str | './tokens_p2p.pkl' | File path for saving/loading extracted tokens |
| `--p2p_use_mask` | int | 0 | Enable mask-guided selective token replacement (0/1) |

### Output

The pipeline generates:

- **source.jpg** - Source image (from source prompt)
- **target.jpg** - Target image (from target prompt with source token guidance)
- **tokens_p2p.pkl** - Extracted tokens for future use

## Technical Details

### Token Extraction (Source Generation)

During source image generation, the pipeline captures bitwise tokens at each scale:

1. Generate indices for current scale via transformer
2. Convert indices to codes via VAE quantizer (`indices_to_codes`)
3. Store codes in BitwiseTokenStorage
4. Continue with normal generation

**Token Format:**
- Shape: `[B, 1, h, w, d]` where:
  - B = batch size
  - h, w = spatial dimensions  
  - d = VAE codebook dimension
- Values: Float tensors (±0.1768 from binary quantization)

### Token Application (Target Generation)

During target image generation, stored tokens are probabilistically applied:

1. Generate indices for current scale via transformer
2. Load corresponding source tokens for this scale
3. Interpolate source tokens to match target spatial size (if needed)
4. Apply probabilistic replacement:
   ```
   mask = rand() < p2p_token_replace_prob
   target_idx = where(mask, source_idx, target_idx)
   ```
5. Convert (possibly modified) indices to codes
6. Continue with normal generation

### Mask-Guided Editing (Future Feature)

The architecture supports optional masks for selective token replacement:

- **Mask Format:** `[B, 1, h, w, 1]` with values in [0, 1]
- **Interpretation:** 1 = use source token, 0 = use target token
- **Application:** `mask = mask & random_mask` for combined control

Example future usage:
```python
# After source generation, compute mask (e.g., via segmentation model)
p2p_storage.save_tokens(scale_idx, tokens, mask=segmentation_mask)

# During target generation, mask automatically applied
gen_one_img(..., p2p_use_mask=True)
```

## BitwiseTokenStorage API

### Initialization

```python
from infinity.utils.bitwise_token_storage import BitwiseTokenStorage

storage = BitwiseTokenStorage(num_scales=5, device='cpu')
```

### Saving Tokens

```python
# Save tokens from a specific scale (typically during source generation)
storage.save_tokens(
    scale_idx=0,
    tokens=idx_tokens,      # Shape: [B, 1, h, w, d]
    mask=segmentation_mask  # Optional: Shape: [B, 1, h, w, 1]
)
```

### Loading Tokens

```python
# Load tokens for application (typically during target generation)
tokens = storage.load_tokens(scale_idx=0, device='cuda')
mask = storage.load_mask(scale_idx=0, device='cuda')
```

### Persistence

```python
# Save to disk
storage.save_to_file('my_tokens.pkl')

# Load from disk
storage.load_from_file('my_tokens.pkl')
```

## Integration with Existing Infinity Pipeline

The P2P pipeline is built on top of existing Infinity components:

- **VAE Quantizer:** Uses `vae.quantizer.lfq.indices_to_codes()` 
- **Scale Schedule:** Works with dynamic resolution system
- **CFG Guidance:** Compatible with classifier-free guidance
- **Rollback Mechanism:** Can be combined with scale-wise rollback

## Examples

### Example 1: Object Replacement

```bash
python3 tools/run_p2p.py \
  --source_prompt "A dog sitting on a chair" \
  --target_prompt "A cat sitting on a chair" \
  --p2p_token_replace_prob 0.7 \
  --save_file ./outputs/dog_to_cat/
```

The high replacement probability (0.7) preserves chair structure while allowing dog→cat semantic change.

### Example 2: Style Transfer with Mask

```bash
# Future: After computing mask for object regions
python3 tools/run_p2p.py \
  --source_prompt "A dog in cartoon style" \
  --target_prompt "A dog in oil painting style" \
  --p2p_token_replace_prob 0.5 \
  --p2p_use_mask 1  # Use saved mask for selective replacement
  --save_file ./outputs/style_transfer/
```

### Example 3: Token Reuse Across Multiple Targets

```bash
# Step 1: Generate and save tokens from source
python3 tools/run_p2p.py \
  --source_prompt "A red car" \
  --target_prompt "A red car" \
  --p2p_token_replace_prob 0.0  # Don't use source tokens yet
  --p2p_token_file ./car_tokens.pkl
  --save_file ./outputs/car_source/

# Step 2: Reuse tokens with different target prompts
python3 tools/run_p2p.py \
  --source_prompt "A red car" \
  --target_prompt "A blue car" \
  --p2p_token_file ./car_tokens.pkl  # Load previously saved tokens
  --p2p_token_replace_prob 0.6
  --save_file ./outputs/car_blue/

# Step 3: Another target
python3 tools/run_p2p.py \
  --source_prompt "A red car" \
  --target_prompt "A red car in rain" \
  --p2p_token_file ./car_tokens.pkl
  --p2p_token_replace_prob 0.5
  --save_file ./outputs/car_rain/
```

## Troubleshooting

### Issue: "BitwiseTokenStorage not found"

**Solution:** Ensure you're importing from the correct path:
```python
from infinity.utils.bitwise_token_storage import BitwiseTokenStorage
```

### Issue: Dimension mismatch errors

**Solution:** The pipeline automatically handles spatial dimension mismatches via interpolation. If you still see errors, check:
- VAE output format matches expected `[B, 1, h, w, d]`
- Scale schedule is consistent with model

### Issue: Mask not being applied

**Solution:** Ensure:
1. Mask was saved during source generation: `storage.save_tokens(..., mask=mask)`
2. Flag is enabled: `--p2p_use_mask 1`
3. Mask values are in [0, 1] range

## Performance Notes

- **Token Storage:** ~500MB for full source generation (5 scales, 1M image)
- **Generation Time:** Same as normal generation (P2P is applied in-place)
- **Memory:** BitwiseTokenStorage stores on CPU; minimal GPU overhead

## Future Extensions

1. **Semantic-Aware Masks:** Auto-generate masks via segmentation models
2. **Advanced Blending:** Weighted blend modes for smoother transitions
3. **Multi-Source Editing:** Blend tokens from multiple source images
4. **Temporal Consistency:** P2P editing for video generation
5. **Region-Specific Control:** Separate replacement probabilities per spatial region

## Related Components

- **Rollback Mechanism:** [ROLLBACK_README.md](../ROLLBACK_README.md)
- **Dynamic Resolution:** `infinity/utils/dynamic_resolution.py`
- **VAE Quantizer:** `infinity/models/bsq_vae/`
- **Infinity Transformer:** `infinity/models/infinity_p2p.py`

## Citation

If you use this P2P pipeline, please cite:

```bibtex
@article{infinity,
  title={Infinity: Scalable Text-to-Image Generation with Cascaded Latent Diffusion},
  ...
}

@article{prompt_to_prompt,
  title={Prompt-to-Prompt Image Editing with Cross-Attention Control},
  ...
}
```
