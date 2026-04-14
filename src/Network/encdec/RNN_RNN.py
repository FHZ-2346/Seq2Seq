import torch
import torch.nn as nn
import torch.nn.functional as F

from Network.EncDec import EncoderDecoder
from Network.Model.RNN import RNNEncoder, RNNDecoder

class RNN_RNN(EncoderDecoder):
    
    def __init__(self, src_vs, tgt_vs, cfg):
        encoder = RNNEncoder(src_vs, cfg)
        decoder = RNNDecoder(tgt_vs, cfg)
        super(RNN_RNN, self).__init__(encoder, decoder)
    
    