import torch
from torch import nn
import numpy as np
import functools


def net_decorator_function(obj):
    class NetWrapper(obj):
        def __init__(self,cfg):
            super().__init__(
                dim_in=cfg.Network.input_dim,
                dim_out=cfg.Network.output_dim,
                n_hid=cfg.Network.n_hid,
                regime=cfg.Network.regime
            )
    return NetWrapper


@net_decorator_function
class Net(nn.Module):
    def __init__(self, dim_in, dim_out, n_hid, regime='rich'):
        super().__init__()
        self.hidden = nn.Linear(dim_in, n_hid)
        self.output = nn.Linear(n_hid, dim_out, bias=False)
        self.activation = nn.ReLU()

        torch.nn.init.normal_(self.hidden.weight, mean=0.0, std=np.sqrt(2 / dim_in))
        torch.nn.init.normal_(self.hidden.bias, mean=0.0, std=np.sqrt(2 / dim_in))
        if regime == 'rich':
            torch.nn.init.normal_(self.output.weight, mean=0.0, std=np.sqrt(2/n_hid))
        elif regime == 'lazy':
            torch.nn.init.normal_(self.output.weight, mean=0.0, std=(2/n_hid))


    def forward(self, x):
        hidden = self.hidden(x)
        hidden = self.activation(hidden)
        out = self.output(hidden)
        return out, hidden

