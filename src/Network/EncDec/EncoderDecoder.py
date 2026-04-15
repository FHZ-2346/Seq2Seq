import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        src: torch.Tensor,
        src_vl: torch.Tensor,
        tgt_with_bos: torch.Tensor,
    ):
        raise NotImplementedError
