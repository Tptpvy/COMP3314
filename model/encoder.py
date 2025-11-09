# model/encoder.py
import torch
import torch.nn as nn

from blocks.encoderBlock import EncoderBlock

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout, max_length, device):
        super(Encoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_mask):
        # x: [batch_size, src_len]
        # src_mask: [batch_size, 1, 1, src_len]
        
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        # Add embeddings
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        # x: [batch_size, src_len, embed_dim]
        
        # Convert mask for attention
        if src_mask is not None:
            src_key_padding_mask = ~src_mask.squeeze(1).squeeze(1).bool()  # [batch_size, src_len]
        else:
            src_key_padding_mask = None
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
            
        return x