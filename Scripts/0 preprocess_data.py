import re
import json
import hydra
from omegaconf import DictConfig

from Data.Utils import tokenize, build_array
from Data.Vocab import Vocab
from Config import cfgG


def save_lines(lines, file_name):
    with open(cfgG.out_dir / file_name, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(" ".join(line) + "\n")


@hydra.main(config_path=f"{cfgG.wsf}/Config", config_name="preprocess", version_base=None)
def main(cfg: DictConfig):
    data_path = cfgG.data_dir / cfg.dataset
    with open(data_path, "r", encoding="utf-8") as f:
        raw = f.read()
    text = raw.replace("\u202f", " ").replace("\xa0", " ").lower()
    text = re.sub(r"(?<! )([,.!?])", r" \1", text)
    src, tgt = tokenize(text, cfg.num_examples)
    save_lines(src, "src.txt")
    save_lines(tgt, "tgt.txt")
    tokens_reserved = ["<pad>", "<bos>", "<eos>"]
    src_vocab = Vocab(src, cfg.min_freq, tokens_reserved)
    tgt_vocab = Vocab(tgt, cfg.min_freq, tokens_reserved)
    vocabs = {"src_i2t": src_vocab.idx2token, "tgt_i2t": tgt_vocab.idx2token}
    with open(cfgG.out_dir / "vocab.json", "w") as f:
        json.dump(vocabs, f)


if __name__ == "__main__":
    main()
