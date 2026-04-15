import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_mask(X, valid_len, invalid_value=0):
    maxlen = X.shape[1]
    valid_mask = torch.arange(maxlen, device=X.device)[None, :] < valid_len[:, None]
    X = X.clone()
    X[~valid_mask] = invalid_value
    return X


class MaskedSoftmax(nn.Module):

    def __init__(self, invalid_value: float = -1e6):
        super().__init__()
        self.invalid_value = invalid_value

    def forward(self, X, valid_lens=None):
        if valid_lens is None:
            return F.softmax(X, dim=-1)
        shape = X.shape  # save original shape, later will restore it
        X_flat = X.reshape(-1, shape[-1])
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X_flat = sequence_mask(X_flat, valid_lens, self.invalid_value)
        X = X_flat.reshape(shape)
        return F.softmax(X, dim=-1)
