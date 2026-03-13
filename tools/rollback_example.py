"""
Example: Using Scale-wise Rollback in Infinity Model

This script demonstrates how to use the rollback mechanism during inference.
The model will generate scales sequentially, but can rollback to previous scales
and regenerate them, enabling self-correction.

Example scenario:
- Normal: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> ... -> 13
- With rollback at scale 4 (1 retry): 
  1 -> 2 -> 3 -> 4 -> [rollback to 3] -> 4 (regenerate) -> 5 -> 6 -> ... -> 13
"""

import torch
from infinity.models import infinity_2b

# Example 1: Rollback scale 4 once
# This will generate: 1, 2, 3, 4, then rollback to 3 and regenerate 4
rollback_schedule_1 = {
    4: 1  # At scale index 4 (0-indexed: so actually the 5th scale), rollback 1 time
}

# Example 2: Multiple rollbacks at different scales
# Scale 3 will be regenerated once, scale 7 will be regenerated twice
rollback_schedule_2 = {
    3: 1,  # Rollback scale 3 once
    7: 2,  # Rollback scale 7 twice
}

# Example 3: Using in actual inference
def generate_with_rollback(model, vae, text_cond, rollback_config):
    """
    Generate images with scale-wise rollback mechanism
    
    Args:
        model: Infinity model
        vae: VAE model
        text_cond: Text conditioning (kv_compact, lens, cu_seqlens_k, max_seqlen_k)
        rollback_config: Dict mapping scale index to number of retries
                        e.g., {4: 1} means rollback scale 4 once
    
    Returns:
        Generated images with self-correction applied
    """
    
    # Define your scale schedule
    scale_schedule = [
        (1, 1, 1),    # scale 0
        (1, 2, 2),    # scale 1
        (1, 3, 3),    # scale 2
        (1, 4, 4),    # scale 3
        (1, 5, 5),    # scale 4 - this will be rolled back if {4: 1}
        (1, 6, 6),    # scale 5
        (1, 8, 8),    # scale 6
        (1, 10, 10),  # scale 7
        (1, 13, 13),  # scale 8
        (1, 16, 16),  # scale 9
        (1, 20, 20),  # scale 10
        (1, 24, 24),  # scale 11
        (1, 30, 30),  # scale 12
    ]
    
    # CFG and temperature schedules
    cfg_list = [1.0] * len(scale_schedule)
    tau_list = [1.0] * len(scale_schedule)
    
    # Call inference with rollback
    with torch.no_grad():
        ret, idx_Bl_list, img = model.autoregressive_infer_cfg(
            vae=vae,
            label_B_or_BLT=text_cond,
            scale_schedule=scale_schedule,
            cfg_list=cfg_list,
            tau_list=tau_list,
            rollback_schedule=rollback_config,  # Enable rollback here!
            ret_img=True,
            B=1,
        )
    
    return img, idx_Bl_list


# Example usage patterns:

# Pattern 1: Simple single rollback
# Generate normally until scale 4, then rollback and regenerate scale 4 once
config_simple = {4: 1}

# Pattern 2: Multiple retries on one scale
# Try scale 6 three times (original + 3 retries = 4 total attempts)
config_multi_retry = {6: 3}

# Pattern 3: Multiple scales with rollback
# This creates a more complex correction pattern
config_complex = {
    3: 1,   # Regenerate scale 3 once
    5: 2,   # Regenerate scale 5 twice  
    8: 1,   # Regenerate scale 8 once
}

# Pattern 4: Progressive refinement
# Rollback early scales to refine the foundation
config_early_refinement = {
    2: 1,  # Refine early stage
    3: 1,  # Refine early stage
    4: 2,  # Multiple refinements at critical scale
}


if __name__ == "__main__":
    print("=" * 60)
    print("Infinity Scale-wise Rollback Examples")
    print("=" * 60)
    
    print("\nExample 1: Single Rollback at Scale 4")
    print("Sequence: 0 -> 1 -> 2 -> 3 -> 4 -> [back to 3] -> 4 -> 5 -> ... -> 12")
    print(f"Config: {config_simple}")
    
    print("\nExample 2: Multiple Retries")
    print("Sequence: ... -> 5 -> 6 -> [back to 5] -> 6 -> [back to 5] -> 6 -> [back to 5] -> 6 -> 7 -> ...")
    print(f"Config: {config_multi_retry}")
    
    print("\nExample 3: Complex Multi-scale Rollback")
    print("Multiple scales will be regenerated at different points")
    print(f"Config: {config_complex}")
    
    print("\nExample 4: Early Stage Refinement")
    print("Focus on refining early scales for better foundation")
    print(f"Config: {config_early_refinement}")
    
    print("\n" + "=" * 60)
    print("How to use in your inference script:")
    print("=" * 60)
    print("""
# In your inference code:
ret, idx_Bl_list, img = model.autoregressive_infer_cfg(
    vae=vae,
    label_B_or_BLT=text_cond,
    scale_schedule=scale_schedule,
    cfg_list=cfg_list,
    tau_list=tau_list,
    rollback_schedule={4: 1},  # <-- Add this parameter!
    ret_img=True,
    B=1,
)
""")
    
    print("\n" + "=" * 60)
    print("Implementation Details:")
    print("=" * 60)
    print("""
1. State Saving: Before each scale in rollback_schedule, the model saves:
   - Hidden states (last_stage)
   - Accumulated codes
   - Current position
   
2. Rollback: After generating the scale, if retries remain:
   - Restore saved state
   - Regenerate the same scale with different random sampling
   
3. Continue: Once all retries are exhausted, continue to next scale

4. Note: This uses different random samples on each retry, so each
   regeneration will be different (assuming top-k/top-p sampling is used)
""")
