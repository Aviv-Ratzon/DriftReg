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
import scipy.io


mpl.rcParams.update({'font.size':10})
mpl.rc('xtick', labelsize=8)
mpl.rc('ytick', labelsize=8)
cmap = mpl.colors.LinearSegmentedColormap.from_list('custom blue', ['#333679','#719ab1', '#f0da49','#ce3723'], N=256)

plt.rcParams['font.sans-serif'] = 'Arial'

fig, axs = plt.subplots(2,2, figsize=(2.2*3.14,1.9*3.14/1.25), sharey=True)


with open('./figures/experimental data/dorgham_data.pkl', 'rb') as handle:
    place_r_mean, place_r_sem, info_mean, info_sem, time_points = pkl.load(handle)

axs[0,0].errorbar(time_points, place_r_mean, yerr=place_r_sem, capsize=3, label='active cells')
# ax2 = plt.gca().twinx()
axs[0,0].errorbar(time_points, info_mean, yerr=info_sem, capsize=3, c='r', label='mean SI')
axs[0,0].set_xlabel('time [m]')
axs[0,0].legend()
# plt.show()
#
# fig.savefig('figures/Fig3/fig_3A.png', dpi=450, format='png', bbox_inches='tight')

Pablo_data = scipy.io.loadmat('./figures/experimental data/Pablo_data.mat')
cell_count = Pablo_data['cell_count']
info = Pablo_data['Info']

cell_count_mean = cell_count.mean(0)
cell_count_mean /= cell_count[:,0].mean()
cell_count_sem = cell_count.std(0)/np.sqrt(cell_count.shape[0]) /  cell_count[:,0].mean()
info_mean = np.array([np.mean(i[0]) for i in info])/np.mean(info[0][0])
info_sem = np.array([np.std(i[0])/np.sqrt(len(i[0][0])) for i in info])/np.mean(info[0][0])

axs[0,1].errorbar(np.arange(len(info)), cell_count_mean, yerr=cell_count_sem, capsize=3)
# ax2 = plt.gca().twinx()
axs[0,1].errorbar(np.arange(len(info)), info_mean, yerr=info_sem, capsize=3, c='r')
axs[0,1].set_xlabel('time [days]')
# plt.show()
#
# fig.savefig('figures/Fig3/fig_3B.png', dpi=450, format='png', bbox_inches='tight')


Lauren_frank_data = scipy.io.loadmat('./figures/experimental data/Lauren_frank_data.mat')
f_active = Lauren_frank_data['f_active']
info = Lauren_frank_data['info_all']

f_active_norm = f_active / f_active[0, :][np.newaxis, :]
f_active_mean = np.nanmean(f_active_norm, 1)
f_active_mean /= f_active_mean[0]
f_active_sem = np.nanstd(f_active_norm,1)/np.sqrt(f_active_norm.shape[1])
info_norm = info / info[0, :][np.newaxis, :]
info_mean = np.nanmean(info_norm, 1)
info_mean /= info_mean[0]
info_sem = np.nanstd(info_norm,1)/np.sqrt(info_norm.shape[1])

axs[1,0].errorbar(np.arange(len(info)), f_active_mean, yerr=f_active_sem, capsize=3)
# ax2 = plt.gca().twinx()
axs[1,0].errorbar(np.arange(len(info)), info_mean, yerr=info_sem, capsize=3, c='r')
axs[1,0].set_xlabel('time [days]')
# plt.show()
#
# fig.savefig('figures/Fig3/fig_3C.png', dpi=450, format='png', bbox_inches='tight')


Sheintuch_data = scipy.io.loadmat('./figures/experimental data/Sheintuch_data.mat')
f_active = Sheintuch_data['frac_active'].T
info = Sheintuch_data['Info'].T

f_active_norm = f_active / f_active[0, :][np.newaxis, :]
f_active_mean = np.nanmean(f_active_norm, 1)
f_active_sem = np.nanstd(f_active_norm,1)/np.sqrt(f_active_norm.shape[1])

info_norm = info / info[0, :][np.newaxis, :]
info_mean = np.nanmean(info_norm, 1)
info_sem = np.nanstd(info_norm,1)/np.sqrt(info_norm.shape[1])

axs[1,1].errorbar(np.arange(len(info)), f_active_mean, yerr=f_active_sem, capsize=3)
# ax2 = plt.gca().twinx()
axs[1,1].errorbar(np.arange(len(info)), info_mean, yerr=info_sem, capsize=3, c='r')
axs[1,1].set_xlabel('time [days]')


axs[0,0].set_ylabel('Normalized value')
axs[1,0].set_ylabel('Normalized value')
plt.subplots_adjust(hspace=0.3)
fig.savefig('figures/Fig3/fig_3.png', dpi=450, format='png', bbox_inches='tight')
plt.show()