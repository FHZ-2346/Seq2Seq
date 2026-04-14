import torch
from torch import nn
from Network.EncDec import Encoder, Decoder


class RNNEncoder(Encoder):

    def __init__(self, src_vs, cfg):
        super(RNNEncoder, self).__init__()
        self.embedding = nn.Embedding(src_vs, cfg.embed_size)
        self.rnn = nn.GRU(cfg.embed_size, cfg.dH, cfg.n_layer, dropout=cfg.dropout)

    def forward(self, X, valid_lens):
        X = self.embedding(X).permute(1, 0, 2)
        output, state = self.rnn(X)
        return output, state


class RNNDecoder(Decoder):

    def __init__(self, tgt_vs, cfg):
        super(RNNDecoder, self).__init__()
        self.embedding = nn.Embedding(tgt_vs, cfg.embed_size)
        self.rnn = nn.GRU(cfg.embed_size + cfg.dH, cfg.dH, cfg.n_layer, dropout=cfg.dropout)
        self.dense = nn.Linear(cfg.dH, tgt_vs)

    def init_state(self, enc_outputs, src_valid_lens):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        return output, state
