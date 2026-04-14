from Network.Model.RNN import RNNEncoder, RNNDecoder
from Network.Model.Attention import Seq2SeqAttentionDecoder
from Network.Model.Transformer.EncDec import TransformerEncoder, TransformerDecoder

enc_dict = {
    "RNN": RNNEncoder,
    "Transformer": TransformerEncoder,
}

dec_dict = {
    "RNN": RNNDecoder,
    "Attention": Seq2SeqAttentionDecoder,
    "Transformer": TransformerDecoder,
}
