import torch
from torch import nn

from .EncoderDecoder import EncoderDecoder
from Network.Module.Attention.Attention import AdditiveAttention
from Network.Enc_Dec.RNN import RNNEncoder, RNNDecoder


class RNNED(EncoderDecoder):

    def __init__(self, src_vs, tgt_vs, cfg):
        encoder = RNNEncoder(src_vs, cfg)
        decoder = RNNDecoder(tgt_vs, cfg)
        super(RNNED, self).__init__(encoder, decoder)
        self.with_attention = cfg.attention
        if self.with_attention:
            self.attention = AdditiveAttention(cfg.dH, cfg.dropout)
        self.dense = nn.Linear(cfg.dH, tgt_vs)

    def forward(self, src, src_vl, tgt_with_bos):
        # Hs: (num_steps, batch_size, dH)
        # S:  (num_layer, batch_size, dH)
        encHs, encS = self.encoder(src, src_vl)

        def get_context(encHs, encS, src_vl):
            num_steps = encHs.shape[0]
            if self.with_attention:
                encS_outmost = encS[-1].unsqueeze(dim=1)
                # (batch_size, 1, dH)
                encHs = encHs.permute(1, 0, 2)
                # (batch_size, num_steps, dH)
                context = self.attention((encS_outmost, encHs, encHs), src_vl)
                context = context.repeat(1, num_steps, 1).permute(1, 0, 2)
            else:
                context = encS[-1].repeat(num_steps, 1, 1)
            return context

        context = get_context(encHs, encS, src_vl)
        decS = encS
        decHs, decS = self.decoder(tgt_with_bos, context, decS)
        tgt_hat = self.dense(decHs.permute(1, 0, 2))
        # (batch_size, num_steps, tgt_vs)
        return tgt_hat

    @torch.no_grad()
    def predict(self, src, src_valid_len, decI, max_len):
        self.eval()
        device = src.device

        # 1) encoder
        encHs, encS = self.encoder(src, src_valid_len)
        decS = encS

        # 2) prepare fixed context
        if self.with_attention:
            encHs = encHs.permute(1, 0, 2)
            context = self.attention(encS[-1].unsqueeze(1), encHs, encHs, src_valid_len)
        else:
            context = encS[-1].unsqueeze(1)   # (batch_size, 1, dH)
        context = context.permute(1, 0, 2) 

        outputs = []
        for _ in range(max_len):
            decO, decS = self.decoder(decI, context, decS)
            pred = decO[:, -1, :].argmax(dim=-1)
            outputs.append(pred)
            decI = pred.unsqueeze(1)

        output_tokens = torch.stack(outputs, dim=1)   # (batch_size, max_len)
        return output_tokens
