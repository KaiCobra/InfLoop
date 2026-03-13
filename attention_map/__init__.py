"""
Attention Map Extraction Module (Block 33 - IQR Filtered)

Self-contained module for extracting IQR-filtered cross-attention maps
from the Infinity transformer model.

Output structure: word × scale (13 scales)

Usage:
    python -m attention_map.run \
        --input exp_prompts/prompts.jsonl \
        --output outputs/attention_map_output
"""

from .extractor import CrossAttentionExtractor

__all__ = ['CrossAttentionExtractor']
