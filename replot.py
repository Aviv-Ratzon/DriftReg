import os
import pandas as pd
import pickle as pkl
from types import SimpleNamespace
import yaml
from yaml.loader import SafeLoader
import numpy as np
from scipy.ndimage import gaussian_filter
from omegaconf import OmegaConf
from utils import plot


name = '/two_envs_lazy'
folder = 'outputs' + name
i=0
# for dir in os.listdir(folder):
path = os.getcwd() + '/' + folder + '/' # + dir + '/'
cfg = OmegaConf.load(path+'.hydra/config.yaml')
os.chdir(path)
if os.path.isfile('output.pkl'):
    plot(cfg)
os.chdir('../../..')
