import torch.nn as nn


class UnifySizeQKV(nn.Module):

    def __init__(self, dH, bias=False):
        super(UnifySizeQKV, self).__init__()
        self.Wq = nn.LazyLinear(dH, bias=bias)
        self.Wk = nn.LazyLinear(dH, bias=bias)
        self.Wv = nn.LazyLinear(dH, bias=bias)

    def forward(self, queries, keys, values):
        return self.Wq(queries), self.Wk(keys), self.Wv(values)
