import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from Network.EncDec import EncoderDecoder
from Network.Model.Transformer.EncDec import TransformerEncoder, TransformerDecoder
from Data.Utils import load_data_nmt
from Utils.Device import try_gpu
from Evaluate import predict_seq2seq, bleu
from Config import cfgG


@hydra.main(config_path=f"{cfgG.wsf}/Config", config_name="Transformer")
def main(cfg: DictConfig):
    device = try_gpu()
    data_iter, src_vocab, tgt_vocab = load_data_nmt(cfg.data, cfg.train.data)

    encoder = TransformerEncoder(cfg)
    decoder = TransformerDecoder(cfg)
    net = EncoderDecoder(encoder, decoder)
    net.to(device)
    net.load_state_dict(torch.load(f"{cfgG.out_dir}/Seq2Seq/Transformer/{net.__class__.__name__}_{cfg.evaluate.load_epoch}.pth"))

    engs = ["go .", "i lost .", "he's calm .", "i'm home ."]
    fras = ["va !", "j'ai perdu .", "il est calme .", "je suis chez moi ."]
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, cfg.train.data.num_steps, device, True
        )
        print(f"{eng} => {translation}, ", f"bleu {bleu(translation, fra, k=2):.3f}")


if __name__ == "__main__":
    main()
