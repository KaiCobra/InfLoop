import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out_proj(context), attn

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        if mask is None:
            mask = torch.tril(torch.ones((seq_len, seq_len))).to(x.device).unsqueeze(0).unsqueeze(1)

        attn_output, _ = self.self_attn(x, x, x, mask=mask)
        x = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_output))
        return x

# --- 測試 Decoder Block ---
batch_size = 2
seq_len = 5
embed_dim = 128
num_heads = 4
ff_dim = 512

decoder_layer = DecoderBlock(embed_dim, num_heads, ff_dim)
x = torch.randn(batch_size, seq_len, embed_dim)
output = decoder_layer(x)
print(f"Decoder Input Shape: {x.shape}")
print(f"Decoder Output Shape: {output.shape}")