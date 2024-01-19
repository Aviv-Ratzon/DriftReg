import hydra
from hydra.core.config_store import ConfigStore
from config import Scenario
from model import Net
from make_data import make_data
from train import Trainer
from analyzer import Analyzer
from utils import plot
import numpy as np
import torch
import time
import os


cs = ConfigStore.instance()
cs.store(name="scenario_config", node=Scenario)

@hydra.main(config_path="conf/files", version_base='1.1')
def main(cfg: Scenario) -> None:
    # if os.path.isfile('plot.html'):
    #     return
    t_start = time.time()
    if cfg.Trainer.seed_fixed:
        np.random.seed(0)
        torch.manual_seed(0)
    # print(omegaconf.OmegaConf.to_yaml(cfg.Trainer))
    make_data(cfg)
    model = Net(cfg).double()
    trainer = Trainer(model, cfg)
    analyzer = Analyzer(trainer, cfg)
    analyzer.run()
    plot(cfg)
    # print(f'Finished run in {round((time.time())-t_start/3600, 2)}\n')


if __name__ == "__main__":
    main()
