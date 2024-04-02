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

df = pd.read_csv('Tables/All.csv')
with open('Tables/All_lines.pkl', 'rb') as handle:
    lines = pkl.load(handle)
active_units_lines = lines['active_units']

# df.drop(index=df[(df.learning_algo == 'SGD') & (df.update_noise == 0)].index, inplace=True)
# df['learning_algo'] = pd.factorize(df['learning_algo'])[0]
# plt.scatter(df['log_tau'], df['active_units']); plt.xlim([-7.5, 0]); plt.show()
#
# fig = plt.figure(figsize=(3.14,3.14/1.5))
# plt.hist(df['active_units'])
# plt.xlim([-0.1,1])
# plt.xlabel('active fraction')
# plt.ylabel('# simulations')
# plt.tight_layout()
# fig.savefig('figures/fig3_A.png', dpi=450, format='png', bbox_inches='tight')
# plt.show()

t = np.linspace(0, df['sim_len'][0], 100)
fig = plt.figure(figsize=(3.14,3.14/2))
active_units_lines_l = np.concatenate([np.array(value) for value in active_units_lines.values()])
plt.plot(t, active_units_lines_l.T, c='k', alpha=0.05, lw=0.5)
plt.plot(t, active_units_lines_l.mean(0), c='tab:red', lw=2)
plt.ylabel('normalized\nfraction active units')
plt.xlabel('training time')
fig.savefig('figures/Fig4/fig4_A.png', dpi=450, format='png', bbox_inches='tight')
plt.show()


df = pd.read_csv('Tables/SGD_update_noise.csv')

# fig = plt.figure(figsize=(3.14,3.14/1.25))
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(3.14,3.14*1.25))
plt.sca(ax1)
x = df['update_noise'].to_numpy(); y = df['active'].to_numpy(); c=df['learning_algo']
x = x[y<1]
y = y[y<1]
# plt.xlabel('$\\sigma^2(\\xi^{update})$')
plt.xticks([])
plt.ylabel('fraction active units')
plt.scatter(x, y, s=4)
plt.tight_layout()
# fig.savefig('figures/noise_vs_active.png', dpi=450, format='png', bbox_inches='tight')
# plt.show()

# fig = plt.figure(figsize=(3.14,3.14/1.25))
plt.sca(ax2)
x = df['update_noise'].to_numpy(); y = df['log_tau'].to_numpy(); c=df['learning_algo']
x = x[y<-4]
y = y[y<-4]
plt.xlabel('$\\sigma^2(\\xi^{update})$')
plt.ylabel('$log(\\tau)$')
plt.scatter(x, y, s=4)
plt.tight_layout()
fig.savefig('figures/Fig4/fig4_3B.png', dpi=450, format='png', bbox_inches='tight')
plt.show()

df = loadmat('outputs/hebbian_results.mat')
t = df['t'].squeeze()
fig = plt.figure(figsize=(3.14,3.14/1.5))
plt.plot(t,df['loss'].squeeze(), zorder=2)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
plt.xlabel('training time')
plt.ylabel('loss')
plt.gca().tick_params(axis='y', colors='#1f77b4')
plt.gca().yaxis.label.set_color('#1f77b4')
ax1 = plt.gca()
ax2 = plt.gca().twinx()
ax2.plot(t,df['actiFrac'].squeeze(), c='r')
ax2.set_ylabel('fraction active units', color='r')
ax2.tick_params(axis='y', colors='red')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.set_ylim([0.3,0.9])

x=ax1.get_xlim()
ax1.axvspan(x[0], 300, facecolor='r', alpha=0.25, zorder=1)
ax1.axvspan(300, 2500, facecolor='y', alpha=0.25, zorder=1)
ax1.axvspan(2500, x[1], facecolor='g', alpha=0.25, zorder=1)

plt.tight_layout()
fig.savefig('figures/Fig4/fig4_3C.png', dpi=450, format='png', bbox_inches='tight')
plt.show()

# corr_mat =df['pvCorr']
# T = len(t)
# T_print = f'$10^{int(np.ceil(np.log10(t[-1])))}$'
# fig = plt.figure(figsize=(3.14,3.14))
# plt.imshow(corr_mat)
# cbar = plt.colorbar(fraction=0.046, pad=0.04)
# cbar.set_label('pearson R')
# plt.xticks([0,T], ['0',T_print])
# plt.gca().xaxis.tick_top()
# plt.xlabel('training time')
# plt.gca().xaxis.set_label_position('top')
# plt.ylabel('training time')
# plt.yticks([0,T], ['0',T_print])
# plt.show()
# fig.savefig('figures/Fig4/fig4_3D.png', dpi=450, format='png', bbox_inches='tight')



# x = [500]
# y = [100]
# for i in range(100000):
#     d_theta = np.array([2*x[-1]*y[-1]**2, 2*x[-1]**2*y[-1]])
#     d_theta = d_theta/np.linalg.norm(d_theta) * 1/1
#     x.append(x[-1] - d_theta[0])
#     y.append(y[-1] - d_theta[1])
# x=np.array(x)
# y= np.array(y)
# X = np.linspace(x.min()-1, x.max()+1, 1000)
# Y = np.linspace(y.min()-1, y.max()+1, 1000)
# X, Y = np.meshgrid(X, Y)
# Z = X**2 * Y**2
# fig = plt.figure(figsize=(2.7, 2.7))
# cs = plt.contourf(X, Y, Z, extend='both')
# plt.plot(x,y)
# plt.show()