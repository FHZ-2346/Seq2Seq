from torch import nn

from .EncoderDecoder import EncoderDecoder
from Network.Enc_Dec.Transformer.Enc_Dec import TransformerEncoder, TransformerDecoder


class TransformerED(EncoderDecoder):

    def __init__(self, src_vs, tgt_vs, cfg):
        encoder = TransformerEncoder(src_vs, cfg)
        decoder = TransformerDecoder(tgt_vs, cfg)
        super(TransformerED, self).__init__(encoder, decoder)
        self.dense = nn.Linear(cfg.dH, tgt_vs)

    def forward(self, src, src_vl, tgt_with_bos):
        encO = self.encoder(src, src_vl)
        # (batch_size, num_steps, dH)
        decO = self.decoder(tgt_with_bos, encO, src_vl)
        # (batch_size, num_steps, dH)
        tgt_hat = self.dense(decO)
        # (batch_size, num_steps, tgt_vs)
        return tgt_hat
