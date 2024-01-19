from torch.optim import SGD, Adam
from model import *
import numpy as np
from config import Scenario
import pickle as pkl
import copy
from train import Trainer
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import calculate_hessian_eigs
from time import time


def analyzer_decorator_function(obj):
    class AnalyzerWrapper(obj):
        def __init__(self,*args, **kwargs):
            super().__init__(*args, **kwargs)
    return AnalyzerWrapper


@analyzer_decorator_function
class Analyzer:
    def __init__(self, trainer: Trainer, cfg: Scenario):
        self.stop_after = cfg.Analyzer.stop_after
        self.converge_sample = cfg.Analyzer.converge_sample
        self.trainer = trainer
        self.num_epoch = int(cfg.Analyzer.num_epoch)
        self.stop_loss = cfg.Trainer.stop_loss
        self.output_path = cfg.Paths.output
        # Network params for plotting
        self.loss_l = np.zeros(int(cfg.Analyzer.num_sample))
        self.loss_start_l = np.zeros(int(1e4+1))
        self.active_l = np.zeros(int(cfg.Analyzer.num_sample))
        self.active_units_l = np.zeros(int(cfg.Analyzer.num_sample))
        self.eigs_l = [None] * int(cfg.Analyzer.num_sample_hess+1)
        self.model_l = []
        # Network params for plotting after mid-way convergance
        self.loss_c_l = np.zeros(int(cfg.Analyzer.num_sample_hess+1))
        self.active_c_l = np.zeros(int(cfg.Analyzer.num_sample_hess+1))
        self.eigs_c_l = [None] * int(cfg.Analyzer.num_sample_hess+1)
        # Sampling freq
        self.sample_epochs_loss = np.linspace(0, 1e4, int(1e4+1)).astype(int)
        self.sample_epochs = np.linspace(0,self.num_epoch-1, int(cfg.Analyzer.num_sample)).astype(int)
        self.sample_epochs_hessian = np.linspace(0,self.num_epoch-1, int(cfg.Analyzer.num_sample_hess)+1).astype(int)

    def run(self):
        t0 = time()
        i = 0
        i_h = 0
        # for epoch in tqdm(range(self.num_epoch)):
        for epoch in (pbar := tqdm(range(self.num_epoch))):
            loss, hidden = self.trainer.run_epoch()
            if epoch in self.sample_epochs:
                loss, hidden = self.trainer.run_clean_forward()
                if loss > 10 or np.isnan(loss) or hidden.sum()==0:
                    print('***** Loss diverged or null solution *****')
                    break
                pbar.set_description(f"Loss = {np.log10(loss):0.2f}")
                self.loss_l[i] = np.log10(loss.detach().numpy())
                self.active_l[i] = (hidden.detach().numpy()>0).mean()
                self.active_units_l[i] = (hidden.detach().numpy().sum(0)>0).mean()
                i += 1
            if epoch in self.sample_epochs_loss:
                loss, hidden = self.trainer.run_clean_forward()
                self.loss_start_l[epoch] = np.log10(loss.detach().numpy())
            if epoch in self.sample_epochs_hessian:
                loss, hidden = self.trainer.run_clean_forward()
                self.eigs_l[i_h] = calculate_hessian_eigs(self.trainer.model, self.trainer.X, self.trainer.y)
                self.model_l.append(copy.deepcopy(self.trainer.model.state_dict()))
                if time()-t0 > self.stop_after*60*60:
                    print('***** Simulation timeout *****')
                    break
                if self.converge_sample:
                    t_c = time()
                    trainer_c = copy.deepcopy(self.trainer)
                    trainer_c.label_noise_scale = 0
                    trainer_c.isotropic_noise_std = 0
                    trainer_c.lr = 0.1
                    trainer_c.optimizer = SGD(trainer_c.model.parameters(), trainer_c.lr)
                    trainer_c.update_weights = trainer_c.pytorch_optimizer
                    patience = 10
                    triggers = 0
                    last_loss, _ = trainer_c.run_epoch()
                    while (loss > 1e-8):
                        loss, hidden = trainer_c.run_epoch()
                        if last_loss < loss:
                            triggers += 1
                        last_loss = loss
                        if (time()-t_c > 5*60) or triggers >= patience:
                            print(f'\ntoo long to converge, triggers = {triggers}, loss = {np.log10(loss.detach().numpy())}')
                            break
                    self.loss_c_l[i_h] = np.log10(loss.detach().numpy())
                    self.active_c_l[i_h] = (hidden.detach().numpy()>0).mean()
                    try:
                        self.eigs_c_l[i_h] = calculate_hessian_eigs(trainer_c.model, trainer_c.X, trainer_c.y)
                    except:
                        self.eigs_c_l[i_h] = np.nan
                i_h += 1
        self.save()

    def save(self):
        with open(self.output_path, 'wb') as handle:
            pkl.dump([{'sample_epochs':self.sample_epochs, 'sample_epochs_hessian':self.sample_epochs_hessian,
                       'loss_l':self.loss_l, 'active_l':self.active_l, 'active_units_l':self.active_units_l, 'eigs_l':self.eigs_l,
                       'loss_c_l':self.loss_c_l, 'active_c_l':self.active_c_l, 'eigs_c_l':self.eigs_c_l, 'model_l':self.model_l,
                       'sample_epochs_loss':self.sample_epochs_loss, 'loss_start_l':self.loss_start_l}], handle)

