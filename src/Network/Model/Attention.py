import torch
from torch import nn
from Network.EncDec import AttentionDecoder
from Network.Module.Attention import AdditiveAttention


class Seq2SeqAttentionDecoder(AttentionDecoder):

    def __init__(self, tgt_vs, cfg):
        super(Seq2SeqAttentionDecoder, self).__init__()
        self.attention = AdditiveAttention(cfg.dH, cfg.dropout)
        self.embedding = nn.Embedding(tgt_vs, cfg.embed_size)
        self.rnn = nn.GRU(cfg.embed_size + cfg.dH, cfg.dH, cfg.n_layer, dropout=cfg.dropout)
        self.dense = nn.Linear(cfg.dH, tgt_vs)

    def init_state(self, enc_outputs, src_valid_lens, *args):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, src_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, src_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        query = torch.unsqueeze(hidden_state[-1], dim=1)
        context = self.attention(query, enc_outputs, enc_outputs, src_valid_lens)
        for x in X: # along time_step
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, src_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
