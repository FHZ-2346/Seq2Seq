import torch
import torch.nn as nn
from .Attention import DotProductAttention
from .UnifySizeQKV import UnifySizeQKV


class MultiHeadAttention(nn.Module):

    def __init__(self, dH, n_head, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.unify_size_qkv = UnifySizeQKV(dH, bias=bias)
        self.n_head = n_head  # dH = n_head * dH_per_head
        self.Wo = nn.Linear(dH, dH, bias=bias)
        self.attention = DotProductAttention(dropout)

    def detach_head_qkv(self, queries, keys, values):

        def detach_head_one(X):
            # (batch_size, seq_len, dH)
            X = X.reshape(X.shape[0], X.shape[1], self.n_head, -1)
            # (batch_size, seq_len, n_head, dH_per_head)
            X = X.permute(0, 2, 1, 3)
            # (batch_size, n_head, seq_len, dH_per_head)
            X = X.reshape(-1, X.shape[2], X.shape[3])
            # (batch_size * n_head, seq_len, dH_per_head)
            return X

        queries = detach_head_one(queries)
        keys = detach_head_one(keys)
        values = detach_head_one(values)
        return queries, keys, values

    def transpose_output(self, X: torch.Tensor):
        # (batch_size * n_head, nq, dH_per_head)
        X = X.reshape(-1, self.n_head, X.shape[1], X.shape[2])
        # (batch_size, n_head, nq, dH_per_head)
        X = X.permute(0, 2, 1, 3)
        # (batch_size, nq, n_head, dH_per_head)
        X = X.reshape(X.shape[0], X.shape[1], -1)
        # (batch_size, nq, dH)
        return X

    def forward(self, qkv, valid_lens):
        queries, keys, values = qkv
        # qkv: (batch_size , n?, d?)
        queries, keys, values = self.unify_size_qkv(queries, keys, values)
        # qkv: (batch_size , n?, dH)
        queries, keys, values = self.detach_head_qkv(queries, keys, values)
        # qkv: (batch_size * n_head, n?, dH_per_head)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.n_head, dim=0)
        attn_output = self.attention((queries, keys, values), valid_lens)
        # (batch_size * n_head, nq, dH_per_head)
        attn_output_concat = self.transpose_output(attn_output)
        # (batch_size, nq, dH)
        output = self.Wo(attn_output_concat)
        # (batch_size, nq, dH)
        return output
