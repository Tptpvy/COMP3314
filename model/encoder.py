import torch
from torch import nn

from blocks.transformerBlock import TransformerBlock

class Encoder(nn.Module):
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
        super(Encoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        # Add word + position embeddings
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        # Convert mask format if needed
        if mask is not None and mask.dim() == 4:
            mask = mask.squeeze(1).squeeze(1)  # [batch_size, src_len]
        
        # Pass through encoder layers
        for layer in self.layers:
            out = layer(out, key_padding_mask=mask)  # Use key_padding_mask for padding
            
        return out