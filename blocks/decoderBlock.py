import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, causal_mask, src_key_padding_mask):
        norm_x = self.norm1(x)
        self_attn, _ = self.self_attention(
            norm_x, norm_x, norm_x,
            attn_mask=causal_mask,
            need_weights=False
        )
        x = x + self.dropout(self_attn)
        
        norm_x = self.norm2(x)
        cross_attn, _ = self.cross_attention(
            norm_x, enc_output, enc_output,
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout(cross_attn)
        
        norm_x = self.norm3(x)
        ff_output = self.ff(norm_x)
        x = x + self.dropout(ff_output)
        
        return x