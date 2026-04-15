import torch
from torch import nn


class RNNEncoder(nn.Module):

    def __init__(self, src_vs, cfg):
        super(RNNEncoder, self).__init__()
        self.embedding = nn.Embedding(src_vs, cfg.embed_size)
        self.rnn = nn.GRU(cfg.embed_size, cfg.dH, cfg.n_layer, dropout=cfg.dropout)

    def forward(self, X, valid_lens):
        X = self.embedding(X).permute(1, 0, 2)
        Hs, state = self.rnn(X)
        return Hs, state


class RNNDecoder(nn.Module):

    def __init__(self, tgt_vs, cfg):
        super(RNNDecoder, self).__init__()
        self.embedding = nn.Embedding(tgt_vs, cfg.embed_size)
        self.rnn = nn.GRU(cfg.embed_size + cfg.dH, cfg.dH, cfg.n_layer, dropout=cfg.dropout)

    def forward(self, X, context, state):
        X = self.embedding(X).permute(1, 0, 2)
        X_and_context = torch.cat((X, context), 2)
        Hs, state = self.rnn(X_and_context, state)
        return Hs, state
