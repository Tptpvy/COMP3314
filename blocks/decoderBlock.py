import torch
from torch import nn

from blocks.transformerBlock import TransformerBlock

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        
        # Self-attention (causal) - uses TransformerBlock
        self.self_attention = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        
        # Cross-attention (encoder-decoder) - uses TransformerBlock
        self.cross_attention = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        
    def forward(self, x, enc_out, trg_mask, src_mask):
        """
        Args:
            x: [batch_size, trg_len, embed_dim]
            enc_out: [batch_size, src_len, embed_dim]
            trg_mask: [batch_size, 1, trg_len, trg_len] (causal mask)
            src_mask: [batch_size, 1, 1, src_len] (padding mask)
        """
        # Convert mask formats
        if trg_mask is not None and trg_mask.dim() == 4:
            trg_mask = trg_mask.squeeze(1)  # [batch_size, trg_len, trg_len]
        
        if src_mask is not None and src_mask.dim() == 4:
            src_mask = src_mask.squeeze(1).squeeze(1)  # [batch_size, src_len]
        
        # Self-attention with causal masking
        x = self.self_attention(x, attn_mask=trg_mask)
        
        # Cross-attention with encoder output
        x = self.cross_attention(
            query=x,
            key=enc_out,
            value=enc_out,
            key_padding_mask=src_mask  # Use source padding mask
        )
        
        return x