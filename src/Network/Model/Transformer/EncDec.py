import torch
import torch.nn as nn
import math
from Network.Module.Attention import MultiHeadAttention
from Network.EncDec import Encoder, AttentionDecoder
from Network.Model.Transformer.Basics import PositionWiseFFN, AddNorm, PositionalEncoding


class EncoderBlock(nn.Module):

    def __init__(self, cfg, use_bias=False):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(cfg.dH, cfg.dH, cfg.MultiHeadAttention.n_head, cfg.dropout, use_bias)
        self.addnorm1 = AddNorm(cfg.AddNorm.norm_shape, cfg.dropout)
        self.ffn = PositionWiseFFN(cfg.dH, cfg.PositionWiseFFN.dH, cfg.dH)
        self.addnorm2 = AddNorm(cfg.AddNorm.norm_shape, cfg.dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(Encoder):

    def __init__(self, src_vs, cfg, use_bias=False):
        super(TransformerEncoder, self).__init__()
        self.dH = cfg.dH
        self.embedding = nn.Embedding(src_vs, cfg.dH)
        self.pos_encoding = PositionalEncoding(self.dH, cfg.dropout)

        self.blks = nn.Sequential()
        for i in range(cfg.n_blk):
            blk = EncoderBlock(cfg, use_bias)
            self.blks.add_module(f"block{i}", blk)

    def forward(self, X, valid_lens):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.dH))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):

    def __init__(self, cfg):
        super(DecoderBlock, self).__init__()
        dH = cfg.dH
        self.attention1 = MultiHeadAttention(dH, dH, cfg.MultiHeadAttention.n_head, cfg.dropout)
        self.addnorm1 = AddNorm(cfg.AddNorm.norm_shape, cfg.dropout)

        self.attention2 = MultiHeadAttention(dH, dH, cfg.MultiHeadAttention.n_head, cfg.dropout)
        self.addnorm2 = AddNorm(cfg.AddNorm.norm_shape, cfg.dropout)
        self.ffn = PositionWiseFFN(dH, cfg.PositionWiseFFN.dH, dH)
        self.addnorm3 = AddNorm(cfg.AddNorm.norm_shape, cfg.dropout)

    def forward(self, X, state, dec_vl):
        encO, src_vl = state
        X2 = self.attention1((X, X, X), dec_vl)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2((Y, encO, encO), src_vl)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z))


class TransformerDecoder(AttentionDecoder):

    def __init__(self, tgt_vs,cfg):
        super(TransformerDecoder, self).__init__()
        self.dH = cfg.dH

        self.embedding = nn.Embedding(tgt_vs, cfg.dH)
        self.pos_encoding = PositionalEncoding(cfg.dH, cfg.dropout)
        self.blks = nn.Sequential()
        for i in range(cfg.n_blk):
            blk = DecoderBlock(cfg)
            self.blks.add_module(f"block{i}", blk)
        self.dense = nn.Linear(cfg.dH, tgt_vs)

    def init_state(self, enc_outputs, src_valid_lens, *args):
        return [enc_outputs, src_valid_lens]

    def forward(self, X, state):
        if self.training:
            batch_size, num_steps = X.shape[0:2]
            dec_vl = torch.arange(1, num_steps + 1, device=X.device)
            dec_vl = dec_vl.repeat(batch_size, 1)
        else:
            dec_vl = None

        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.dH))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X = blk(X, state, dec_vl)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
