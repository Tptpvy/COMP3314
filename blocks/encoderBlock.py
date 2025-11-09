import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, key_padding_mask=None):
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attention(
            norm_x, norm_x, norm_x, 
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_output)
        
        norm_x = self.norm2(x)
        ff_output = self.ff(norm_x)
        x = x + self.dropout(ff_output)
        
        return x