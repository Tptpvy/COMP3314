# model/decoder.py
import math
import torch
import torch.nn as nn
from blocks.decoderBlock import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout, max_length, device):
        super(Decoder, self).__init__()
        self.device = device
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, trg_mask, src_mask):
        """
        Args:
            x: [batch_size, trg_len]
            enc_output: [batch_size, src_len, embed_dim]
            trg_mask: [batch_size, 1, trg_len, trg_len]
            src_mask: [batch_size, 1, 1, src_len]
        """
        N, seq_length = x.shape
        
        # Create positions directly on the correct device
        positions = torch.arange(0, seq_length, device=self.device).expand(N, seq_length)
        
        # Add embeddings
        x = self.word_embedding(x) * math.sqrt(self.embed_dim)  # Scale embeddings
        x = self.dropout(x + self.position_embedding(positions))
        # x: [batch_size, trg_len, embed_dim]
        
        # Convert masks
        if trg_mask is not None:
            # For causal mask: [trg_len, trg_len]
            # Create directly on device
            causal_mask = torch.triu(
                torch.ones(seq_length, seq_length, device=x.device) * float('-inf'), 
                diagonal=1
            )
        else:
            causal_mask = None
            
        if src_mask is not None:
            src_key_padding_mask = ~src_mask.squeeze(1).squeeze(1).bool()  # [batch_size, src_len]
        else:
            src_key_padding_mask = None
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x, enc_output, causal_mask, src_key_padding_mask)
            
        return x