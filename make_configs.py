import hydra
from hydra.core.config_store import ConfigStore
from config import Scenario, _DATATYPE, _ALGOTYPE
from typing import get_args
import numpy as np
from utils import varname, update_cfg
from omegaconf import OmegaConf,open_dict


cs = ConfigStore.instance()
cs.store(name="scenario_config", node=Scenario)


@hydra.main(config_path="conf", config_name="config", version_base='1.1')
def main(cfg: Scenario) -> None:
    cfg.Analyzer.converge_sample = False
    cfg.Analyzer.stop_after = 10 # hours
    data_types = get_args(_DATATYPE) # ['smoothed'] #
    algo_types = ['SGD']#get_args(_ALGOTYPE)
    lrs =  [1e-1] #np.logspace(-2,-1,3)#
    X_dims = [20,25,30]
    y_dims = [5, 10, 15]
    n_samples = [5,10,20]
    noise_scales = [0.03] # np.linspace(1e-1,1e0,30)
    # n_hids = [100]
    # X_dims = [int(cfg.Network.n_hid/5)]
    # n_samples = [int(cfg.Network.n_hid/20)]
    regimes = ['rich', 'lazy']
    config_num = 0
    for data_type in data_types:
        for algo_type in algo_types:
            for lr in lrs:
                for X_dim in X_dims:
                    for regime in regimes:
                        for y_dim in y_dims:
                            for n_sample in n_samples:
                                cfg.Data.type = data_type
                                cfg.Trainer.Algorithm = algo_type
                                cfg.Trainer.lr = float(lr)
                                cfg.Network.regime = regime
                                cfg.Data.X_dim = X_dim
                                cfg.Data.y_dim = y_dim
                                cfg.Data.n_samples = int(n_sample)
                                if algo_type in ['RLgrad', 'Adam']:
                                    cfg.Trainer.label_noise = 0
                                    cfg.Trainer.update_noise = 0
                                    f = open(f"../../../conf/files/config{config_num}.yaml", "w")
                                    f.write(OmegaConf.to_yaml(cfg))
                                    config_num += 1
                                elif algo_type == 'SGD':
                                    for noise_scale in noise_scales:
                                        cfg.Trainer.label_noise = float(1) #float(noise_scale)
                                        cfg.Trainer.update_noise = 0
                                        f = open(f"../../../conf/files/config{config_num}.yaml", "w")
                                        f.write(OmegaConf.to_yaml(cfg))
                                        config_num += 1
                                    for noise_scale in noise_scales:
                                        cfg.Trainer.label_noise = 0
                                        cfg.Trainer.update_noise = float(0.03) # float(noise_scale)
                                        f = open(f"../../../conf/files/config{config_num}.yaml", "w")
                                        f.write(OmegaConf.to_yaml(cfg))
                                        config_num += 1
    print(f'Created {config_num} config files')




if __name__ == "__main__":
    main()
