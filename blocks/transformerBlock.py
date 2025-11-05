import torch
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, attn_mask=None, key_padding_mask=None):
        """
        Handles both self-attention and cross-attention
        - For self-attention: key, value = None (uses query)
        - For cross-attention: provide key and value
        """
        # Default to self-attention if key/value not provided
        if key is None:
            key = query
        if value is None:
            value = query
            
        # Convert key_padding_mask to correct format if needed
        if key_padding_mask is not None and key_padding_mask.dim() == 4:
            key_padding_mask = key_padding_mask.squeeze(1).squeeze(1)  # [N, seq_len]
            
        # Attention
        attn_output, _ = self.attention(
            query, key, value, 
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        
        # Add & Norm
        x = query + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward network
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x