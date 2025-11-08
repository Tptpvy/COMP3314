import torch
from torch import nn

from blocks.decoderBlock import DecoderBlock

class Decoder(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embed_dim, 
        num_heads, 
        ff_dim, 
        num_layers, 
        dropout,
        max_length,
        device
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    # def forward(self, x, enc_out, trg_mask, src_mask):
    #     N, seq_length = x.shape
    #     positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
    #     # Add word + position embeddings
    #     out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
    #     # Pass through decoder layers
    #     for layer in self.layers:
    #         out = layer(out, enc_out, trg_mask, src_mask)
            
    #     return out
    
    def forward(self, x, enc_out, trg_mask, src_mask):
        """
        Args:
            x: [batch_size, trg_len, embed_dim]
            enc_out: [batch_size, src_len, embed_dim]
            trg_mask: [batch_size, 1, trg_len, trg_len] (causal mask)
            src_mask: [batch_size, 1, 1, src_len] (padding mask)
        """
        # Self-attention with causal masking
        x = self.self_attention(x, attn_mask=trg_mask)
        
        # Cross-attention with encoder output
        # Use source padding mask for cross-attention
        x = self.cross_attention(
            query=x,
            key=enc_out,
            value=enc_out,
            key_padding_mask=src_mask.squeeze(1).squeeze(1) if src_mask is not None else None
        )
        
        return x