import torch
import torch.nn as nn
import math
from blocks.encoderBlock import EncoderBlock
from embedding.positional_encoding import SinusoidalPositionalEncoding

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout, max_length, device):
        super(Encoder, self).__init__()
        self.device = device
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_length)
        
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_mask):
        N, seq_length = x.shape
        
        x = self.word_embedding(x) * math.sqrt(self.embed_dim)
        x = self.dropout(self.pos_encoding(x))
        
        if src_mask is not None:
            src_key_padding_mask = ~src_mask.squeeze(1).squeeze(1).bool()
        else:
            src_key_padding_mask = None
        
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
            
        return x