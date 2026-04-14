import json

from Data.Vocab import Vocab
from Data.Utils import build_array, unify_array, load_data_iter
from Config import cfgG


def load_data_nmt(cfg):
    with open(cfgG.out_dir / "src.txt", "r", encoding="utf-8") as f:
        src_lines = [line.strip().split() for line in f]
    with open(cfgG.out_dir / "tgt.txt", "r", encoding="utf-8") as f:
        tgt_lines = [line.strip().split() for line in f]

    with open(cfgG.out_dir / "vocab.json", "r") as f:
        vocab_dict = json.load(f)
    src_vocab, tgt_vocab = Vocab(), Vocab()
    src_vocab.load_map_i2t(vocab_dict["src_i2t"])
    tgt_vocab.load_map_i2t(vocab_dict["tgt_i2t"])

    src_arrays = build_array(src_lines, src_vocab)
    tgt_arrays = build_array(tgt_lines, tgt_vocab)

    src_arrays_unified, src_valid_len = unify_array(src_arrays, src_vocab, cfg.num_steps)
    tgt_arrays_unified, tgt_valid_len = unify_array(tgt_arrays, tgt_vocab, cfg.num_steps)
    data_arrays = (src_arrays_unified, src_valid_len, tgt_arrays_unified, tgt_valid_len)
    data_iter = load_data_iter(data_arrays, cfg.batch_size, is_train=True)
    return data_iter, src_vocab, tgt_vocab
