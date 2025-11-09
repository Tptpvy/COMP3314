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
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        dropout=0.1,
        device="cpu",
        max_length=100
    ):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads, d_ff, num_layers, 
            dropout, max_length, device
        )
        self.decoder = Decoder(
            trg_vocab_size, d_model, num_heads, d_ff, num_layers,
            dropout, max_length, device
        )
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.fc_out = nn.Linear(d_model, trg_vocab_size)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        
        return trg_mask
        
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, trg_mask, src_mask)
        
        output = self.fc_out(dec_output)
        return output