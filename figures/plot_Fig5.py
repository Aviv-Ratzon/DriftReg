import hydra
from hydra.core.config_store import ConfigStore
from config import Scenario
from model import Net
import os
from utils import *
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
import matplotlib as mpl
from matplotlib import font_manager
from scipy.stats import pearsonr
import yaml
from yaml.loader import SafeLoader
import pandas as pd
from scipy.io import loadmat


mpl.rcParams.update({'font.size':10})
mpl.rc('xtick', labelsize=8)
mpl.rc('ytick', labelsize=8)
cmap = mpl.colors.LinearSegmentedColormap.from_list('custom blue', ['#333679','#719ab1', '#f0da49','#ce3723'], N=256)

plt.rcParams['font.sans-serif'] = 'Arial'

plt.jet()


df = pd.read_csv('Tables/SGD_update_noise.csv')

fig = plt.figure(figsize=(3.14,3.14/1.5))
x = df['eig_nonzero'].to_numpy(); y = df['active_units'].to_numpy()
plt.xlabel('fraction non-zero eigenvalues')
plt.ylabel('fraction active units')
plt.scatter(x, y, s=4)
plt.tight_layout()
fig.savefig('figures/Fig5/fig5_B.png', dpi=450, format='png', bbox_inches='tight')
plt.show()


folder = "outputs/predictive_track_up/"
cfg, output_dict, d = retrieve_data(folder)

fig = plt.figure(figsize=(3.14,3.14))
eigs = output_dict['eigs_l']
eigs1 = np.log10(eigs[20])
eigs2 = np.log10(eigs[30])
ind = np.isfinite(eigs1) & np.isfinite((eigs2))
eigs1 = eigs1[ind]
eigs2 = eigs2[ind]
lims = [np.min([eigs1.min(),eigs2.min()]), np.max([eigs1.max(),eigs2.max()])]
plt.plot(lims, lims, c='gray', ls='--', lw=5, zorder=1)
plt.scatter(eigs1, eigs2, c='purple', zorder=2, s=5)
# plt.scatter(np.log10(eigs[20]), np.log10(eigs[30]), c='purple', zorder=2, s=5)
# plt.scatter(np.log10(eigs[70]), np.log10(eigs[90]), c='purple', zorder=2, s=5)
plt.xlabel('$log(\lambda^{t})$')
plt.ylabel('$log(\lambda^{t+\Delta t})$')
plt.tight_layout()
plt.show()
fig.savefig('figures/Fig5/fig5_C.png', dpi=450, format='png', bbox_inches='tight')


folder = "outputs/update_and_label_compare/10/"
cfg, output_dict, d = retrieve_data(folder)

fig = plt.figure(figsize=(3.14, 3.14))
plt.plot(output_dict['sample_epochs_hessian'], [e.sum() for e in output_dict['eigs_l']], c='purple', lw=4)
plt.xlabel('training time')
plt.ylabel('$\sum{\lambda_i}$')
# ax2.set_ylim([0,1])
plt.tight_layout()
plt.show()
fig.savefig('figures/Fig5/fig5_D.png', dpi=450, format='png', bbox_inches='tight')