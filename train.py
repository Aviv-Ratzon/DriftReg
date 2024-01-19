from torch.optim import SGD, Adam
from model import *
import numpy as np
from config import Scenario
import pickle as pkl
import copy


class Trainer:
    def __init__(self, model: Net, cfg: Scenario):
        with open(cfg.Paths.data, 'rb') as handle:
            data = pkl.load(handle)
        self.X = data['X']
        self.y = data['y']
        self.y_var = self.y.var()
        self.loss_fn = torch.nn.MSELoss()
        self.input_noise_scale = cfg.Trainer.input_noise
        self.label_noise_scale = cfg.Trainer.label_noise
        self.isotropic_noise_std = cfg.Trainer.update_noise
        self.model = model
        self.lr = cfg.Trainer.lr
        if cfg.Trainer.Algorithm == 'SGD':
            self.optimizer = SGD(model.parameters(), self.lr)
            self.update_weights = self.pytorch_optimizer
        elif cfg.Trainer.Algorithm == 'Adam':
            self.optimizer = Adam(model.parameters(), self.lr)
            self.update_weights = self.pytorch_optimizer
        elif cfg.Trainer.Algorithm == 'RLgrad':
            self.model_new = copy.deepcopy(self.model)
            self.update_weights = self.RLgrad_optimizer

    def run_clean_forward(self):
        with torch.no_grad():
            out, hidden = self.model(self.X)
            loss = self.loss_fn(out, self.y)
            return loss/self.y_var, hidden

    def run_epoch(self):
        out, hidden = self.model(self.X + torch.normal(mean=0, std=self.input_noise_scale, size=self.X.shape))
        y_noise = self.y + (torch.rand(size=self.y.shape) - 0.5) * self.label_noise_scale
        loss = self.loss_fn(out, y_noise)
        self.update_weights(loss)
        return loss/self.y_var, hidden

    def pytorch_optimizer(self, loss):
        loss.backward()
        for W in self.model.parameters():
            W.grad += torch.normal(mean=0, std=self.isotropic_noise_std, size=W.shape)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def RLgrad_optimizer(self, loss):
        with torch.no_grad():
            for W, W_new in zip(self.model.parameters(), self.model_new.parameters()):
                Wp = torch.normal(mean=0, std=0.1, size=W.shape) / np.sqrt(W.shape[0])
                W_new.set_(W + Wp)
            out, _ = self.model_new(self.X)
            L_samp = self.loss_fn(out, self.y)
            for W, W_new in zip(self.model.parameters(), self.model_new.parameters()):
                grad = (W_new - W) * (L_samp - loss)
                W.set_(W - self.lr * grad)


