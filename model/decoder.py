import torch
import torch.nn as nn
import math
from blocks.decoderBlock import DecoderBlock
from embedding.positional_encoding import SinusoidalPositionalEncoding

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout, max_length, device):
        super(Decoder, self).__init__()
        self.device = device
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_length)
        
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, trg_mask, src_mask):
        N, seq_length = x.shape
        
        x = self.word_embedding(x) * math.sqrt(self.embed_dim)
        x = self.dropout(self.pos_encoding(x))
        
        if trg_mask is not None:
            causal_mask = torch.triu(
                torch.ones(seq_length, seq_length, device=x.device) * float('-inf'), 
                diagonal=1
            )
        else:
            causal_mask = None
            
        if src_mask is not None:
            src_key_padding_mask = ~src_mask.squeeze(1).squeeze(1).bool()
        else:
            src_key_padding_mask = None
        
        for layer in self.layers:
            x = layer(x, enc_output, causal_mask, src_key_padding_mask)
            
        return x