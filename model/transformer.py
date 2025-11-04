import torch
from torch import nn

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
        ff_dim=1024,
        num_layers=6,
        dropout=0.1,
        device="cuda",
        max_length=100
    ):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            src_vocab_size, 
            embed_dim, 
            num_heads, 
            ff_dim, 
            num_layers, 
            dropout,
            max_length,
            device
        )
        self.decoder = Decoder(
            trg_vocab_size, 
            embed_dim, 
            num_heads, 
            ff_dim, 
            num_layers, 
            dropout,
            max_length,
            device
        )
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
        # Add projection layer to output vocabulary size
        self.fc_out = nn.Linear(embed_dim, trg_vocab_size)

    def make_src_mask(self, src):
        # Create padding mask for encoder (src_mask: [N, 1, 1, src_len])
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        # Create causal mask for decoder (trg_mask: [N, 1, trg_len, trg_len])
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, trg_mask, src_mask)
        
        # Project to vocabulary size
        output = self.fc_out(dec_output)
        return output