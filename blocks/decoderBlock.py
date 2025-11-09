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
            nn.GELU(),  # Keeping GELU
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, causal_mask, src_key_padding_mask):
        # Post-LayerNorm self-attention
        self_attn, _ = self.self_attention(
            x, x, x,
            attn_mask=causal_mask,
            need_weights=False
        )
        x = self.norm1(x + self.dropout(self_attn))  # Post-LN
        
        # Post-LayerNorm cross-attention
        cross_attn, _ = self.cross_attention(
            x, enc_output, enc_output,
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )
        x = self.norm2(x + self.dropout(cross_attn))  # Post-LN
        
        # Post-LayerNorm feed forward
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))  # Post-LN
        
        return x