import os
import pandas as pd
import pickle as pkl
from types import SimpleNamespace
import yaml
from yaml.loader import SafeLoader
import numpy as np
from scipy.ndimage import gaussian_filter
from utils import *
from tqdm import tqdm

mpl.rcParams.update({'font.size':10})
mpl.rc('xtick', labelsize=8)
mpl.rc('ytick', labelsize=8)
cmap = mpl.colors.LinearSegmentedColormap.from_list('custom blue', ['#333679','#719ab1', '#f0da49','#ce3723'], N=256)


df = pd.read_csv('Tables/All.csv')
with open('Tables/All_lines.pkl', 'rb') as handle:
    lines_dicts = pkl.load(handle)
t = np.linspace(0, df['sim_len'][0], 100)

active_units_lines = lines_dicts['active_units']
active_lines = lines_dicts['active']

fig, axs = plt.subplots(2,3, figsize=(7,11/3))
for i, (key, value) in enumerate(active_units_lines.items()):
    active_units_lines[key] = np.array(value)
    axs[0,i].plot(t, active_units_lines[key].T, c='k', alpha=0.1)
    axs[0,i].plot(t, np.mean(active_units_lines[key],0), c='tab:red', lw=2)
    axs[0,i].set_ylim([-0.1,1.1])
    axs[0,i].set_xticks([])
for i, (key, value) in enumerate(active_lines.items()):
    active_lines[key] = np.array(value)
    axs[1,i].plot(t, active_lines[key].T, c='k', alpha=0.1)
    axs[1,i].plot(t, np.mean(active_lines[key],0), c='tab:red', lw=2)
    axs[1,i].set_ylim([-0.1,1.1])

axs[0,1].set_yticks([]); axs[0,2].set_yticks([])
axs[1,1].set_yticks([]); axs[1,2].set_yticks([])
axs[0,0].set_title('SED')
axs[0,1].set_title('SGD')
axs[0,2].set_title('Adam')
axs[-1,0].set_xlabel('training time')
axs[-1,1].set_xlabel('training time')
axs[-1,2].set_xlabel('training time')
axs[0,0].set_ylabel('fraction active units')
axs[1,0].set_ylabel('active fraction')
plt.tight_layout()
fig.savefig('figures/FigS1/figS1.png', dpi=450, format='png', bbox_inches='tight')
plt.show()