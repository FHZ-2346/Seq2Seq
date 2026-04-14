import torch
from torch import device, nn
from tqdm import tqdm

from Network.EncDec import EncoderDecoder
from Network.EncDec_dict import enc_dict, dec_dict
from Train.Utils import grad_clipping, xavier_init_weights
from Train.Loss import MaskedSoftmaxCELoss
from Utils.Auxiliary import Timer, Accumulator
from Config import cfgG


class TrainPipeline:

    def __init__(self, cfg, src_vocab, tgt_vocab, device):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        encoder = enc_dict[cfg.network.encoder](len(src_vocab), cfg.network)
        decoder = dec_dict[cfg.network.decoder](len(tgt_vocab), cfg.network)
        self.net = EncoderDecoder(encoder, decoder)
        self.net.apply(xavier_init_weights)
        self.net.to(device)

        self.loss = MaskedSoftmaxCELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.train.lr)
        self.device = device
        self.metric = Accumulator(2)

    def train_epoch(self, train_iter):
        for batch in train_iter:
            src, src_vl, tgt, tgt_vl = [x.to(self.device) for x in batch]
            batch_size = src.shape[0]
            bos = self.tgt_vocab["<bos>"]
            bos = torch.tensor([bos] * batch_size, device=self.device).reshape(-1, 1)
            tgt_with_bos = torch.cat([bos, tgt[:, :-1]], dim=1)
            tgt_hat = self.net(encI=(src, src_vl), decI=tgt_with_bos)

            self.optimizer.zero_grad()
            l = self.loss(tgt_hat, tgt, tgt_vl)
            l.sum().backward()
            grad_clipping(self.net, 1)
            self.optimizer.step()

            with torch.no_grad():
                num_tokens = tgt_vl.sum()
                self.metric.add(l.sum(), num_tokens)

    def train(self, train_iter, cfgT, cfgN):
        self.net.train()
        pbar = tqdm(range(cfgT.num_epochs))
        for epoch in pbar:
            timer = Timer()
            self.train_epoch(train_iter)
            if (epoch + 1) % 10 == 0:
                pbar.set_description(f"epoch {epoch + 1}, loss {self.metric[0] / self.metric[1]:.3f}, {self.metric[1] / timer.stop():.1f} tokens/sec on {str(self.device)}")
            if (epoch + 1) % cfgT.save_freq == 0:
                save_dir = cfgG.checkpoint_dir / f"{cfgN.encoder}+{cfgN.decoder}"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{epoch + 1}.pth"
                torch.save(self.net.state_dict(), save_path)
