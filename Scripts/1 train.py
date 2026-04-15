import hydra
from omegaconf import DictConfig

from Data.Load import load_data_nmt
from Utils.Device import try_gpu
from Train.TrainPipeline import TrainPipeline
from Config import cfgG


@hydra.main(
    config_path=f"{cfgG.wsf}/Config",
    config_name="RNN",  # Transformer, RNN
    version_base=None,
)
def main(cfg: DictConfig):
    data_iter, src_vocab, tgt_vocab = load_data_nmt(cfg.train.data)
    train_pip = TrainPipeline(cfg, src_vocab, tgt_vocab, try_gpu())
    train_pip.train(data_iter, cfg.train, cfg.network)

if __name__ == "__main__":
    main()
