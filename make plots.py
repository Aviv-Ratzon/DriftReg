import hydra
from hydra.core.config_store import ConfigStore
from config import Scenario
from model import Net
from utils import *
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
import matplotlib as mpl
from matplotlib import font_manager
from scipy.io import loadmat


fpath = 'bahnschrift.ttf'
font_manager.fontManager.addfont(fpath)
prop = font_manager.FontProperties(fname=fpath)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams.update({'font.size':20})
mpl.rc('xtick', labelsize=15)
mpl.rc('ytick', labelsize=15)

plt.rcParams['font.sans-serif'] = 'Calibry'


vars = loadmat('hebb_results.mat')
fig = plt.figure(figsize=(10, 5))
plt.plot(vars['t'].T, vars['loss'])
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.xlabel('Time')
plt.ylabel('Similarity matching')
ax2 = plt.gca().twinx()
ax2.plot(vars['t'].T, vars['actiFrac'].T, c='r')
ax2.set_ylabel('Active fraction', color='r')
ax2.tick_params(axis='y', colors='red')
# ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax2.set_ylim([0,1])
plt.tight_layout()
plt.show()
fig.savefig('figures/loss_vs_active_hebb.png', dpi=450, format='png')