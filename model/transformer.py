# model/transformer.py
import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_dim=256,
        num_heads=8,
        ff_dim=512,
        num_layers=3,
        dropout=0.1,
        device="cpu",
        max_length=100
    ):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            src_vocab_size, embed_dim, num_heads, ff_dim, num_layers, 
            dropout, max_length, device
        )
        self.decoder = Decoder(
            trg_vocab_size, embed_dim, num_heads, ff_dim, num_layers,
            dropout, max_length, device
        )
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.fc_out = nn.Linear(embed_dim, trg_vocab_size)

    def make_src_mask(self, src):
        # src: [batch_size, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask: [batch_size, 1, 1, src_len]
        return src_mask

    def make_trg_mask(self, trg):
        # trg: [batch_size, trg_len]
        batch_size, trg_len = trg.shape
        
        # Padding mask
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask: [batch_size, 1, 1, trg_len]
        
        # Causal mask
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_sub_mask: [trg_len, trg_len]
        
        # Combine masks
        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask: [batch_size, 1, trg_len, trg_len]
        
        return trg_mask
        
    def forward(self, src, trg):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_output = self.encoder(src, src_mask)  # [batch_size, src_len, embed_dim]
        dec_output = self.decoder(trg, enc_output, trg_mask, src_mask)  # [batch_size, trg_len, embed_dim]
        
        # Project to vocabulary size
        output = self.fc_out(dec_output)  # [batch_size, trg_len, trg_vocab_size]
        return output