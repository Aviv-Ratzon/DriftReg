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

np.seterr(all='raise')

def plot_shaded(ax, t, y, color, label):
    N = y.shape[0]
    ax.fill_between(t, y.mean(0)-y.std(0), y.mean(0)+y.std(0), color=color, alpha=0.1, ec=None)
    ax.plot(t, y.mean(0), c=color, label=label, lw=0.5)


fig,axs = plt.subplots(2,2, figsize=(10,8))
folder_name = 'update_and_label_compare'
folder = 'outputs/' + folder_name
fields = ['path', 'loss', 'active', 'active_units', 'eigs_sum', 'log_eigs_sum', 'eigs_max', 'eigs_nonzero']
lines_dicts = {}
for i, name in enumerate(['update_noise', 'label_noise']):
    lines_dict = {k: [] for k in fields}
    for dir in tqdm(os.listdir(folder)):
        path = os.getcwd() + '/' + folder + '/' + dir + '/'
        if os.path.isfile(path + 'output.pkl'):
            with open(path + '.hydra/config.yaml') as f:
                cfg = yaml.load(f, Loader=SafeLoader)
            if cfg['Trainer'][name] == 0 or cfg['Trainer']['Algorithm'] != 'SGD':
                continue
            with open(path + 'output.pkl', 'rb') as handle:
                data_dict = pkl.load(handle)[0]
            d = SimpleNamespace(**data_dict)
            if any(d.active_l == 0) or (len(d.eigs_l)<100):
                continue
            log_eigs_sum = np.array([np.sum(np.log10(eigs[eigs>0])) for eigs in d.eigs_l])
            eigs_sum = np.array([np.sum(eigs) for eigs in d.eigs_l])
            eigs_max = np.array([np.max(eigs) for eigs in d.eigs_l])
            eigs_nonzero = np.array([np.mean(eigs>0) for eigs in d.eigs_l])
            lines_dict['path'].append(path)
            lines_dict['loss'].append(d.loss_l)
            lines_dict['active'].append(d.active_l/d.active_l[0])
            lines_dict['active_units'].append(d.active_units_l/d.active_units_l[0])
            lines_dict['log_eigs_sum'].append(log_eigs_sum/log_eigs_sum[0])
            lines_dict['eigs_sum'].append(eigs_sum/eigs_sum[0])
            lines_dict['eigs_max'].append(eigs_max/eigs_max[0])
            lines_dict['eigs_nonzero'].append(eigs_nonzero/eigs_nonzero[0])
    for key in lines_dict.keys():
        lines_dict[key] = np.array(lines_dict[key])

    t = d.sample_epochs
    t_h = d.sample_epochs_hessian
    plot_shaded(axs[0, 0], t, lines_dict['active'], ['blue', 'orange'][i], name)
    plot_shaded(axs[0, 1], t, lines_dict['active_units'], ['blue', 'orange'][i], name)
    plot_shaded(axs[1, 0], t_h, lines_dict['eigs_sum'], ['blue', 'orange'][i], name)
    plot_shaded(axs[1, 1], t_h, lines_dict['eigs_nonzero'], ['blue', 'orange'][i], name)
    # axs[2, 0].plot(lines_dict['eigs_max'].mean(0))
    # axs[2, 0].plot(lines_dict['eigs_max'].T, c=['blue', 'orange'][i], alpha=0.1, lw=0.5)
    # axs[2, 1].plot(lines_dict['log_eigs_sum'].mean(0))
    # axs[2, 1].plot(lines_dict['log_eigs_sum'].T, c=['blue', 'orange'][i], alpha=0.1, lw=0.5)
    lines_dicts[name] = lines_dict
print(lines_dicts['update_noise']['active'].shape[0]+lines_dicts['label_noise']['active'].shape[0])
axs[0,0].legend()
axs[0,0].set_title('active')
axs[0,1].set_title('active units')
axs[1,0].set_title('eigs sum')
axs[1,0].set_yscale("log")
axs[1,1].set_title('eigs nonzero')
# axs[2,0].set_title('eigs max')
# axs[2,1].set_title('log eigs sum')
plt.show()

lines_dicts['t'] = t
lines_dicts['t_h'] = t_h

with open('Tables/' + folder_name + '_lines_dicts.pkl', 'wb') as handle:
    pkl.dump(lines_dicts, handle)
