import hydra
from hydra.core.config_store import ConfigStore
from config import Scenario, _DATATYPE, _ALGOTYPE
from typing import get_args
import numpy as np
from utils import varname, update_cfg
from omegaconf import OmegaConf,open_dict
import random


cs = ConfigStore.instance()
cs.store(name="scenario_config", node=Scenario)


@hydra.main(config_path="conf", config_name="config", version_base='1.1')
def main(cfg: Scenario) -> None:
    cfg.Analyzer.converge_sample = False
    cfg.Analyzer.stop_after = 10 # hours
    data_types = get_args(_DATATYPE) # ['smoothed'] #
    data_types = [d for d in data_types if d!='predictive_track']
    algo_types = ['RLgrad'] # get_args(_ALGOTYPE)
    regimes = ['rich', 'lazy']
    lrs = {'Adam':np.linspace(0.5e-3,1.5e-3,100), 'SGD':np.linspace(0.05,0.15,100), 'RLgrad':np.linspace(0.9,1,100)}
    X_dims = np.arange(5,30)
    y_dims = np.arange(5,30)
    n_hid = [50,100,150,250,500]
    n_samples = np.arange(5,15)
    noise_scales_update = np.logspace(-2,-1,50)
    noise_scales_label = np.logspace(-1,0,50)
    size_x = np.arange(1,10)
    size_y = np.arange(5,15)
    kernel = np.arange(0.01,0.1,0.01)
    FOV_angle_frac = [1,2,4,8,16]
    V0 = np.arange(0.1,1,0.1)
    n_configs = 100
    cfg.Network.n_hid = 100
    for config_num in range(n_configs):
        cfg.Data.type = random.choice(data_types)
        cfg.Trainer.update_noise = 0
        cfg.Trainer.label_noise = 0
        cfg.Trainer.Algorithm = random.choice(algo_types)
        cfg.Trainer.lr = float(random.choice(lrs[cfg.Trainer.Algorithm]))
        cfg.Network.regime = random.choice(regimes)
        # cfg.Network.n_hid = random.choice(n_hid)
        cfg.Data.y_dim = int(random.choice(y_dims))
        cfg.Data.X_dim = int(random.choice(X_dims))
        cfg.Data.n_samples = int(random.choice(n_samples))
        if cfg.Data.type=='predictive_track':
            cfg.Env.size_x = int(random.choice(size_x))
            cfg.Env.size_y = int(random.choice(size_y))
            cfg.Env.kernel = float(random.choice(kernel))
            cfg.Env.FOV_angle_frac = int(random.choice(FOV_angle_frac))
            cfg.Env.V0 = float(random.choice(V0))

        if cfg.Trainer.Algorithm == 'SGD':
            if random.choice([False, True]):
                cfg.Trainer.label_noise = float(random.choice(noise_scales_label))
            else:
                cfg.Trainer.update_noise = float(random.choice(noise_scales_update))
        f = open(f"../../../conf/files/config{config_num}.yaml", "w")
        f.write(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
