import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.alias import *


class Classifier(nn.Module):

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions.
        Defined in :numref:`sec_classification`"""
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = astype(argmax(Y_hat, axis=1), Y.dtype)
        compare = astype(preds == reshape(Y, -1), torch.float32)
        return reduce_mean(compare) if averaged else compare

    def loss(self, Y_hat, Y, averaged=True):
        """Defined in :numref:`sec_softmax_concise`"""
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = reshape(Y, (-1,))
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')

    def layer_summary(self, X_shape):
        """Defined in :numref:`sec_lenet`"""
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

    def init_state(self, enc_outputs, src_valid_lens):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(Classifier):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encI, decI):
        src, src_vl = encI
        encO = self.encoder(src, src_vl)
        tgt_with_bos = decI
        dec_state = self.decoder.init_state(encO, src_vl)
        return self.decoder(tgt_with_bos, dec_state)[0]

    def predict_step(self, batch, device, num_steps, save_attention_weights=False):
        """Defined in :numref:`sec_seq2seq_training`"""
        batch = [to(a, device) for a in batch]
        src, tgt, src_valid_len, _ = batch
        enc_all_outputs = self.encoder(src, src_valid_len)
        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
        outputs = [torch.expand_dims(tgt[:, 0], 1)]
        attention_weights = []
        for _ in range(num_steps):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
            outputs.append(argmax(Y, 2))
            # Save attention weights (to be covered later)
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        return torch.cat(outputs[1:], 1), attention_weights


class AttentionDecoder(Decoder):

    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
