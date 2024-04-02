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


mpl.rcParams.update({'font.size':10})
mpl.rc('xtick', labelsize=8)
mpl.rc('ytick', labelsize=8)
cmap = mpl.colors.LinearSegmentedColormap.from_list('custom blue', ['#333679','#719ab1', '#f0da49','#ce3723'], N=256)

plt.rcParams['font.sans-serif'] = 'Arial'

folder = "outputs/predictive_track_up/"

plt.jet()
with open(folder + '.hydra/config.yaml') as f:
    cfg = yaml.load(f, Loader=SafeLoader)
cfg = cfg_to_namespace(cfg)
with open(folder + cfg.Paths.output, 'rb') as handle:
    output_dict = pkl.load(handle)[0]
with open(folder + cfg.Paths.data, 'rb') as handle:
    data_dict = pkl.load(handle)
d = SimpleNamespace(**output_dict)
t = output_dict['sample_epochs_hessian']
T = len(t)
T_print = f'$10^{int(np.ceil(np.log10(t[-1])))}$'
X = data_dict['X']; y = data_dict['y']; pos = data_dict['pos']

cfg.Network.input_dim = output_dict['model_l'][0]['hidden.weight'].shape[1]
cfg.Network.output_dim = output_dict['model_l'][0]['output.weight'].shape[0]
model = Net(cfg).double()

infos_l, rate_maps_l, is_place_l = calc_rate_maps(output_dict['model_l'], pos, model, X)

corr_mat = np.zeros([T,T])
for t1 in tqdm(range(T)):
    for t2 in range(t1+1,T):
        cells = np.where((is_place_l[t1]) & (is_place_l[t2]))[0]
        corr_mat[t1,t2] = np.nanmean([pearsonr(rate_maps_l[t1][cell,:], rate_maps_l[t2][cell,:])[0] for cell in cells])
corr_mat += corr_mat.T
corr_mat[np.arange(T), np.arange(T)] = 1
fig = plt.figure(figsize=(3.14,3.14))
plt.imshow(corr_mat)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.set_label('pearson R')
plt.xticks([0,T], ['0',T_print])
plt.gca().xaxis.tick_top()
plt.xlabel('training time')
plt.gca().xaxis.set_label_position('top')
plt.ylabel('training time')
plt.yticks([0,T], ['0',T_print])
plt.show()
fig.savefig('figures/Fig2/fig_2E.png', dpi=450, format='png', bbox_inches='tight')

fig = plot_rate_maps(cfg, folder=folder)
# plt.tight_layout()
plt.show()
fig.savefig('figures/Fig2/fig_2D.png', dpi=450, format='png', bbox_inches='tight')

info_l = []
active_l = []
active_units_l = []
for i in range(int(T)):
    model.load_state_dict(output_dict['model_l'][i])
    out, hidden = model(X)
    out = out.detach().numpy(); hidden = hidden.detach().numpy()
    infos, rate_maps, is_place = calc_rate_map(pos,hidden)
    info_l.append(infos[is_place].mean())
    active_l.append((hidden>0).mean())
    active_units_l.append((hidden.sum(0)>0).mean())
# t = t[:int(T/2)]
fig = plt.figure(figsize=(3.6,1.6))
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(3.6, 2*1.35))
plt.sca(ax2)
plt.plot(t,info_l)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.xlabel('training time')
plt.ylabel('mean SI')

plt.gca().tick_params(axis='y', colors='#1f77b4')
plt.gca().yaxis.label.set_color('#1f77b4')
ax2 = plt.gca().twinx()
ax2.plot(t, active_l, c='r')
ax2.set_ylabel('fraction active units', color='r')
ax2.tick_params(axis='y', colors='red')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.tight_layout()
plt.sca(ax1)
T_loss = int(len(output_dict['sample_epochs']))
loss_norm = np.log10(((y-y.mean(0)[np.newaxis,:])**2).mean()).numpy()
plt.plot(output_dict['sample_epochs'][:T_loss],output_dict['loss_l'][:T_loss]-loss_norm, c='r', lw=1)
plt.ylabel('log loss')
plt.xticks([])
plt.tight_layout()
plt.show()
fig.savefig('figures/Fig2/fig_2BC.png', dpi=450, format='png', bbox_inches='tight')



