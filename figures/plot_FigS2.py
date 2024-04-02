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


def plot_shaded(ax, t, y, color, label):
    N = y.shape[0]
    ax.fill_between(t, y.mean(0)-y.std(0), y.mean(0)+y.std(0), color=color, alpha=0.1, ec=None)
    ax.plot(t, y.mean(0), c=color, label=label, lw=0.5)
    # ax.plot(t, y.T, c=color, label=label, lw=0.5, alpha=0.1)


with open('Tables/update_and_label_compare_lines_dicts.pkl', 'rb') as handle:
    lines_dicts = pkl.load(handle)

fig,axs = plt.subplots(2,2, figsize=(3.14*2,2.8*2))
for i, name in tqdm(enumerate(['update_noise', 'label_noise'])):
    lines_dict = lines_dicts[name]
    t = lines_dicts['t']
    t_h = lines_dicts['t_h']
    plot_shaded(axs[0, 0], t, lines_dict['active'], ['blue', 'orange'][i], ['update noise', 'label noise'][i])
    plot_shaded(axs[0, 1], t, lines_dict['active_units'], ['blue', 'orange'][i], name)
    plot_shaded(axs[1, 0], t_h, lines_dict['eigs_sum'], ['blue', 'orange'][i], name)
    plot_shaded(axs[1, 1], t_h, lines_dict['eigs_nonzero'], ['blue', 'orange'][i], name)

axs[0,0].legend()
axs[0,0].set_ylabel('active fraction')
axs[0,1].set_ylabel('fraction active units')
axs[1,0].set_ylabel('$\sum \lambda_i$')
axs[1,1].set_ylabel('fraction non-zero eigenvalues')

axs[1,0].set_xlabel('training time')
axs[1,1].set_xlabel('training time')

axs[0,0].set_xticks([])
axs[0,1].set_xticks([])
# axs[0,1].set_yticks([])

axs[1,0].set_yscale("log")
# axs[2,0].set_title('eigs max')
# axs[2,1].set_title('log eigs sum')
fig.tight_layout(pad=2)
fig.savefig('figures/FigS2/fig_S2.png', dpi=450, format='png', bbox_inches='tight')
plt.show()
