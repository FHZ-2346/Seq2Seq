from Data.Download import read_data_nmt
from Data.Dataset.NMT import tokenize_nmt, preprocess_nmt, build_array_nmt, load_array
from Data.Vocab import Vocab
from Config import cfgT

raw = read_data_nmt()
text = preprocess_nmt(raw)
src, tgt = tokenize_nmt(text, num_examples=100)

min_freq = 2
tokens_reserved = ["<pad>", "<bos>", "<eos>"]
src_vocab = Vocab(src, min_freq, tokens_reserved)
tgt_vocab = Vocab(tgt, min_freq, tokens_reserved)

src_array, src_valid_len = build_array_nmt(src, src_vocab, cfgT.num_steps)
tgt_array, tgt_valid_len = build_array_nmt(tgt, tgt_vocab, cfgT.num_steps)

# Test
src_arr = src_array[0]
src_tokens = src_vocab.to_tokens(src_arr)
print("tokens:", src_tokens)
print("array:", src_arr)
print("valid length:", src_valid_len[0])
