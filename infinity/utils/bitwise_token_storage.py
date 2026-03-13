"""
BitwiseTokenStorage: Efficient storage and retrieval for VAE quantizer indices from image generation.

This class handles:
1. Saving quantizer indices (0/1 integers) from source image generation
2. Loading and applying indices during target image generation
3. Support for optional mask-based token selection (for future mask-guided editing)

Token Format:
- Shape: [B, 1, h, w, d] where d is the VAE codebook dimension
- Values: Integer tensors (0 or 1) representing quantized indices
- Stored on CPU to save GPU memory
"""

import torch
import os
from typing import Dict, List, Optional, Tuple
import pickle


class BitwiseTokenStorage:
    """Storage container for bitwise tokens from source prompt generation."""
    
    def __init__(self, num_scales: int = 5, device: str = 'cpu'):
        """
        Initialize BitwiseTokenStorage.
        
        Args:
            num_scales: Number of scales to store tokens for (e.g., 5 means first 5 scales)
            device: Device to store tokens on (default 'cpu' for memory efficiency)
        """
        self.num_scales = num_scales
        self.device = device
        self.tokens: Dict[int, torch.Tensor] = {}  # scale_idx -> tokens [B,1,h,w,d]
        self.masks: Dict[int, torch.Tensor] = {}   # scale_idx -> mask [B,1,h,w,1] (optional)
        self.scale_shapes: Dict[int, Tuple] = {}   # scale_idx -> (B, h, w, d)
        
    def save_tokens(self, scale_idx: int, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        """
        Save bitwise tokens from a specific scale.
        
        Args:
            scale_idx: Index of the scale (0-indexed)
            tokens: Token tensor [B, 1, h, w, d] with values ±0.1768
            mask: Optional mask tensor [B, 1, h, w, 1] for future mask-guided editing
                 Values should be in [0, 1] where 1 means use source token
        """
        if scale_idx >= self.num_scales:
            return  # Ignore scales beyond our storage limit
        
        # Store on CPU to save GPU memory
        self.tokens[scale_idx] = tokens.detach().cpu()
        self.scale_shapes[scale_idx] = (tokens.shape[0], tokens.shape[2], tokens.shape[3], tokens.shape[4])
        
        if mask is not None:
            self.masks[scale_idx] = mask.detach().cpu()
    
    def load_tokens(self, scale_idx: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Load tokens for a specific scale and move to target device.
        
        Args:
            scale_idx: Index of the scale
            device: Target device to move tokens to
            
        Returns:
            Token tensor [B, 1, h, w, d] or None if not available
        """
        if scale_idx not in self.tokens:
            return None
        
        return self.tokens[scale_idx].to(device)
    
    def load_mask(self, scale_idx: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Load mask for a specific scale.
        
        Args:
            scale_idx: Index of the scale
            device: Target device
            
        Returns:
            Mask tensor [B, 1, h, w, 1] or None if not available
        """
        if scale_idx not in self.masks:
            return None
        
        return self.masks[scale_idx].to(device)
    
    def has_tokens_for_scale(self, scale_idx: int) -> bool:
        """Check if tokens are available for a given scale."""
        return scale_idx in self.tokens
    
    def has_mask_for_scale(self, scale_idx: int) -> bool:
        """Check if mask is available for a given scale."""
        return scale_idx in self.masks
    
    def get_num_stored_scales(self) -> int:
        """Get number of scales with stored tokens."""
        return len(self.tokens)
    
    def clear(self) -> None:
        """Clear all stored tokens and masks."""
        self.tokens.clear()
        self.masks.clear()
        self.scale_shapes.clear()
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save all tokens and masks to disk.
        
        Args:
            filepath: Path to save to (e.g., 'tokens.pkl')
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        data = {
            'tokens': self.tokens,
            'masks': self.masks,
            'scale_shapes': self.scale_shapes,
            'num_scales': self.num_scales,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"BitwiseTokenStorage saved to {filepath} ({len(self.tokens)} scales)")
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load tokens and masks from disk.
        
        Args:
            filepath: Path to load from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Token file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.tokens = data['tokens']
        self.masks = data['masks']
        self.scale_shapes = data['scale_shapes']
        self.num_scales = data['num_scales']
        
        print(f"BitwiseTokenStorage loaded from {filepath} ({len(self.tokens)} scales)")
    
    def __repr__(self) -> str:
        """String representation showing stored scales."""
        scales_str = ', '.join([f"scale_{i}: {tuple(self.scale_shapes[i])}" 
                                for i in sorted(self.tokens.keys())])
        masks_str = ', '.join([f"scale_{i}" for i in sorted(self.masks.keys())])
        return (f"BitwiseTokenStorage(num_scales={self.num_scales}, "
                f"stored_scales=[{scales_str}], "
                f"masks=[{masks_str}], device='{self.device}')")
