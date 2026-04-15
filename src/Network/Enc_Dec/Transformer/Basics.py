import torch
import torch.nn as nn


class PositionWiseFFN(nn.Module):

    def __init__(self, dI, dH, dO):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dI, dH),
            nn.ReLU(),
            nn.Linear(dH, dO),
        )

    def forward(self, X):
        return self.net(X)


class AddNorm(nn.Module):

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        assert X.shape == Y.shape
        return self.ln(self.dropout(Y) + X)


class PositionalEncoding(nn.Module):

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)
