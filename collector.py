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

name = 'SGD_update_noise'
folder = 'outputs/' # + name
cols = ['data_type', 'regime', 'Xdim', 'ydim', 'nsamples', 'learning_algo', 'lr', 'update_noise', 'label_noise',
            'loss', 'active', 'active_units', 'log_tau', 'eig_sum', 'eig_max', 'eig_nonzero', 'link', 'sim_len']
df = pd.DataFrame(columns=cols)
plot_fs = []
active_units_lines = {'RLgrad':[], 'SGD':[], 'Adam':[]}
active_lines = {'RLgrad':[], 'SGD':[], 'Adam':[]}
for pdir in ['SGD_update_noise']: # ['random', 'update_and_label_compare', 'SGD_vary_update_noise', 'RLgrad_random']: #['']: #  # ['RLgrad_random']:#
    print('\n' + pdir)
    for dir in tqdm(os.listdir(folder+pdir)):
        path = os.getcwd() + '/' + folder + '/' + pdir + '/' + dir + '/'
        if not os.path.isfile(path + 'output.pkl'):
            # print(f'skipped {pdir}/{dir} because no output.pkl')
            continue
        with open(path + 'output.pkl', 'rb') as handle:
            data_dict = pkl.load(handle)[0]
        d = SimpleNamespace(**data_dict)
        with open(path + '.hydra/config.yaml') as f:
            cfg = yaml.load(f, Loader=SafeLoader)

        if not hasattr(d, 'active_units_l'):
            continue

        smoothed_loss = d.loss_l[np.linspace(0, len(d.loss_l) - 1, 1000).astype(int)]
        smoothed_loss = gaussian_filter(smoothed_loss, sigma=len(d.loss_l) / 100)

        if any(d.active_l == 0) or (len(d.eigs_l)<100) or (d.loss_l[-1]>10) or (pdir=='random' and cfg['Trainer']['Algorithm']=='RLgrad'):
            continue
        plot_fs.append(path + 'plot.html')
        loss = d.loss_l
        # i_end_c = np.where([l is not None for l in d.eigs_l])[0][-1]
        # i_end = np.where(d.loss_l!=0)[0][-1]
        # i_start = 1
        # d.loss_l = d.loss_l[i_start:i_end]; d.active_l = d.active_l[i_start:i_end]; d.eigs_l = d.eigs_l[i_start:i_end_c]
        # d.sample_epochs_hessian = d.sample_epochs_hessian[i_start:i_end]; d.sample_epochs = d.sample_epochs[i_start:i_end]

        try:
            tau = np.log10(abs(get_tau(d.sample_epochs, d.active_units_l)))
        except:
            tau = 0

        active_units = d.active_units_l[-11:-1].mean()/d.active_units_l[0]

        i_start_h = abs(d.sample_epochs_hessian-d.sample_epochs[0]).argmin()
        eigs_sum = np.array([np.sum(eigs) if eigs is not None else np.nan for eigs in d.eigs_l])
        eigs_max = np.array([np.max(eigs) if eigs is not None else np.nan for eigs in d.eigs_l])
        eigs_nonzero = np.array([np.mean(eigs>0) if eigs is not None else np.nan for eigs in d.eigs_l])
        df.loc[len(df)] = [cfg['Data']['type'], cfg['Network']['regime'],
                           cfg['Data']['X_dim'], cfg['Data']['y_dim'], cfg['Data']['n_samples'],
                           cfg['Trainer']['Algorithm'], cfg['Trainer']['lr'], cfg['Trainer']['update_noise'],
                           cfg['Trainer']['label_noise'], d.loss_l[-1], d.active_l[-11:-1].mean()/d.active_l[0],
                           active_units,tau,
                           eigs_sum[-11:-1].mean()/eigs_sum[0:10].mean(),
                           eigs_max[-11:-1].mean()/eigs_max[0:10].mean(),
                           eigs_nonzero[-11:-1].mean(),
                           path + '\plot.html', d.sample_epochs[-1]]
        if not (cfg['Trainer']['Algorithm'] == 'SGD' and cfg['Trainer']['label_noise'] > 0):
            active_units_l = d.active_units_l/d.active_units_l[0]
            active_units_l = active_units_l[np.linspace(0,len(active_units_l)-1, 100).astype(int)]
            active_units_lines[cfg['Trainer']['Algorithm']].append(active_units_l)
            active_l = d.active_l/d.active_l[0]
            active_l = active_l[np.linspace(0,len(active_l)-1, 100).astype(int)]
            active_lines[cfg['Trainer']['Algorithm']].append(active_l)
df.to_csv('Tables/' + name + '.csv', index=False)
with open('Tables/' + name + '_lines.pkl', 'wb') as handle:
    pkl.dump({'active_units':active_units_lines, 'active':active_lines}, handle)

# fig, axs = plt.subplots(3,2)
# for i, (key, value) in enumerate(active_units_lines.items()):
#     active_units_lines[key] = np.array(value)
#     axs[i,0].plot(active_units_lines[key].T, c='k', alpha=0.1)
#     axs[i,0].set_title(key)
# for i, (key, value) in enumerate(active_lines.items()):
#     active_lines[key] = np.array(value)
#     axs[i,1].plot(active_lines[key].T, c='k', alpha=0.1)
#     axs[i,1].set_title(key)
# plt.show()
# c = ['red', 'blue', 'green']
# for i, (key, value) in enumerate(active_units_lines.items()):
#     active_units_lines[key] = np.array(value)
#     plt.plot(active_units_lines[key].T, c=c[i], alpha=0.1)
# plt.show()