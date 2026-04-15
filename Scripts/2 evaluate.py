import torch
from torch import nn
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from Network.EncDec_dict import encdec_dict
from Data.Load import load_data_nmt
from Utils.Device import try_gpu
from Evaluate import predict_seq2seq, bleu, process_src
from Config import cfgG


@hydra.main(config_path=f"{cfgG.wsf}/Config", config_name="RNN")
def main(cfg: DictConfig):
    device = try_gpu()
    encdec_name = f"{cfg.network.encoder}+{cfg.network.decoder}"

    data_iter, src_vocab, tgt_vocab = load_data_nmt(cfg.train.data)
    net: nn.Module = encdec_dict[encdec_name](len(src_vocab), len(tgt_vocab), cfg.network)
    net.to(device)
    save_dir = cfgG.checkpoint_dir / f"{cfg.network.encoder}+{cfg.network.decoder}"
    net.load_state_dict(torch.load(save_dir / f"{cfg.evaluate.load_epoch}.pth"))

    engs = ["go .", "i lost .", "he's calm .", "i'm home ."]
    fras = ["va !", "j'ai perdu .", "il est calme .", "je suis chez moi ."]
    for eng, fra in zip(engs, fras):
        src, src_valid_len = process_src(eng, src_vocab, cfg.train.data.num_steps, device)
        decI = torch.unsqueeze(torch.tensor([tgt_vocab["<bos>"]], dtype=torch.long, device=device), dim=0)
        tgt = net.predict(src, src_valid_len, decI, cfg.train.data.num_steps)
        translation = tgt_vocab.to_tokens(tgt[0])
        print(f"{eng} => {translation}")
        pass
        # translation, dec_attention_weight_seq = predict_seq2seq(
        #     net, eng, src_vocab, tgt_vocab, cfg.train.data.num_steps, device, True
        # )
        # print(f"{eng} => {translation}, ", f"bleu {bleu(translation, fra, k=2):.3f}")


if __name__ == "__main__":
    main()
