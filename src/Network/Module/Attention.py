import torch
import torch.nn as nn
import math

def masked_softmax(X, valid_lens):

    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class AdditiveAttention(nn.Module):

    def __init__(self, d_hid, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(d_hid, bias=False)
        self.W_q = nn.LazyLinear(d_hid, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):

    def __init__(self, size, d_hid, n_head, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        if type(size) == tuple:
            key_size, query_size, value_size = size
        elif type(size) == int: # 自注意力
            key_size = query_size = value_size = size

        self.n_head = n_head
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, d_hid, bias=bias)
        self.W_k = nn.Linear(key_size, d_hid, bias=bias)
        self.W_v = nn.Linear(value_size, d_hid, bias=bias)
        self.W_o = nn.Linear(d_hid, d_hid, bias=bias)

    def transpose_qkv(self, X: torch.Tensor):
        X = X.reshape(X.shape[0], X.shape[1], self.n_head, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X: torch.Tensor):
        X = X.reshape(-1, self.n_head, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, inputs, valid_lens):
        if type(inputs) == tuple:
            queries, keys, values = inputs
        else:
            assert type(inputs) == torch.Tensor
            queries = inputs.clone()
            keys = inputs.clone()
            values = inputs.clone()
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.n_head, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
