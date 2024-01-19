from torch import matmul, relu, cat, real
from torch.linalg import eig
from torch.autograd.functional import hessian
import pickle as pkl
from types import SimpleNamespace
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from config import Scenario
import json
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import warnings
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic
from model import Net
import matplotlib as mpl
import yaml
from yaml.loader import SafeLoader


mpl.rcParams.update({'font.size':10})
mpl.rc('xtick', labelsize=8)
mpl.rc('ytick', labelsize=8)
cmap = mpl.colors.LinearSegmentedColormap.from_list('custom blue', ['#333679', '#719ab1', '#f0da49', '#ce3723'], N=256)

warnings.simplefilter(action='ignore', category=FutureWarning)


def calculate_hessian_eigs(model, x, y):
    Din = model.hidden.weight.shape[1]
    Dout = model.output.weight.shape[0]
    N = model.hidden.weight.shape[0]
    n = model.output.weight.T
    m = model.hidden.weight.T
    b = model.hidden.bias

    yhat = matmul(relu(x.matmul(m) + b), n)
    L = ((yhat - y) ** 2).mean()
    L.backward()

    def def_Loss(x, y, Din, Dout, N):
        def Loss(theta):
            n = theta[:N * Dout].reshape([N, Dout])
            m = theta[N * Dout:N * Dout + Din * N].reshape([Din, N])
            b = theta[N * Dout + Din * N:].reshape([1, N])
            return ((matmul(relu(x.matmul(m) + b), n) - y) ** 2).mean()

        return Loss

    loss_f = def_Loss(x, y, Din, Dout, N)

    hess = hessian(loss_f, cat([n.flatten(), m.flatten(), b.flatten()]), vectorize=True)
    res = eig(hess)
    eigs = real(res[0])
    return eigs.numpy()


def calculate_hessian_eigs_v(model, x, y):
    Din = model.hidden.weight.shape[1]
    Dout = model.output.weight.shape[0]
    N = model.hidden.weight.shape[0]
    n = model.output.weight.T
    m = model.hidden.weight.T
    b = model.hidden.bias

    yhat = matmul(relu(x.matmul(m) + b), n)
    L = ((yhat - y) ** 2).mean()
    L.backward()

    def def_Loss(x, y, Din, Dout, N):
        def Loss(theta):
            n = theta[:N * Dout].reshape([N, Dout])
            m = theta[N * Dout:N * Dout + Din * N].reshape([Din, N])
            b = theta[N * Dout + Din * N:].reshape([1, N])
            return ((matmul(relu(x.matmul(m) + b), n) - y) ** 2).mean()

        return Loss

    loss_f = def_Loss(x, y, Din, Dout, N)

    hess = hessian(loss_f, cat([n.flatten(), m.flatten(), b.flatten()]), vectorize=True)
    res = eig(hess)
    eigs = real(res[0])
    eigs_v = real(res[1])
    return eigs.numpy(), eigs_v.numpy(), cat(
        [n.flatten(), m.flatten(), b.flatten()]).detach().numpy(), hess.detach().numpy()


def plot(cfg: Scenario):
    with open(cfg.Paths.output, 'rb') as handle:
        data_dict = pkl.load(handle)[0]
    d = SimpleNamespace(**data_dict)

    button = list([
        dict(
            args=[{"xaxis.type": "linear"}],
            label="Linear Scale",
            method="relayout"
        ),
        dict(
            args=[{"xaxis.type": "log"}],
            label="Log Scale",
            method="relayout"
        ),
        dict(
            args=[{"xaxis4.type": "linear"}],
            label="Linear Scale",
            method="relayout"
        ),
        dict(
            args=[{"xaxis4.type": "log"}],
            label="Log Scale",
            method="relayout"
        )
    ])

    updatemenus = [
        dict(
            type="buttons",
            direction="down",
            buttons=button
        ),
    ]
    fig = make_subplots(rows=6, cols=3)

    fig.update_layout(updatemenus=updatemenus)
    txt = json.dumps(OmegaConf.to_object(cfg)).replace(',', '<br>').replace('\'', '').replace('{', '').replace('}',
                                                                                                               '').replace(
        '\"', '')
    txt = txt
    fig.add_annotation(
        x=3, y=1,
        xref='x3',
        yref='y3 domain',
        align='left',
        valign='bottom',
        text=txt,
        font={'size': 20},
        showarrow=False
    )

    fig.add_trace(
        go.Line(x=d.sample_epochs, y=d.loss_l),
        row=1, col=1
    )

    fig.add_trace(
        go.Line(x=d.sample_epochs, y=d.active_l),
        row=2, col=1
    )

    fig.add_trace(
        go.Line(x=d.sample_epochs, y=d.active_units_l),
        row=3, col=1
    )

    try:
        fig.add_trace(
            go.Line(x=d.sample_epochs_hessian, y=[np.sum(eigs) if eigs is not None else np.nan for eigs in d.eigs_l]),
            row=4, col=1
        )

        fig.add_trace(
            go.Line(x=d.sample_epochs_hessian, y=[np.max(eigs) if eigs is not None else np.nan for eigs in d.eigs_l]),
            row=5, col=1
        )

        fig.add_trace(
            go.Line(x=d.sample_epochs_hessian,
                    y=[np.mean(eigs > 0) if eigs is not None else np.nan for eigs in d.eigs_l]),
            row=6, col=1
        )
    except:
        print('Nones in eigs')

    if cfg.Analyzer.converge_sample:
        fig.add_trace(
            go.Line(x=d.sample_epochs_hessian, y=d.loss_c_l),
            row=1, col=2
        )

        fig.add_trace(
            go.Line(x=d.sample_epochs_hessian, y=d.active_c_l),
            row=2, col=2
        )

        fig.add_trace(
            go.Line(x=d.sample_epochs_hessian, y=[np.sum(eigs) if eigs is not None else np.nan for eigs in d.eigs_c_l]),
            row=3, col=2
        )

        fig.add_trace(
            go.Line(x=d.sample_epochs_hessian,
                    y=[np.sum(np.log(eigs[eigs > 0])) if eigs is not None else np.nan for eigs in d.eigs_c_l]),
            row=4, col=2
        )

        fig.add_trace(
            go.Line(x=d.sample_epochs_hessian, y=[np.max(eigs) if eigs is not None else np.nan for eigs in d.eigs_c_l]),
            row=5, col=2
        )

        fig.add_trace(
            go.Line(x=d.sample_epochs_hessian,
                    y=[np.mean(eigs > 0) if eigs is not None else np.nan for eigs in d.eigs_c_l]),
            row=6, col=2
        )

    # Update xaxis properties
    fig.update_xaxes(title_text="epoch", row=1, col=1)
    fig.update_xaxes(title_text="epoch", row=2, col=1)
    fig.update_xaxes(title_text="epoch", row=3, col=1)
    fig.update_xaxes(title_text="epoch", row=4, col=1)
    fig.update_xaxes(title_text="epoch", row=5, col=1)
    fig.update_xaxes(title_text="epoch", row=6, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="loss", row=1, col=1)
    fig.update_yaxes(title_text="fraction active", range=[-0.1, 1.1], row=2, col=1)
    fig.update_yaxes(title_text="fraction active units", range=[-0.1, 1.1], row=3, col=1)
    fig.update_yaxes(title_text="sum of eigenvalues", row=4, col=1)
    fig.update_yaxes(title_text="max eigenvalue", row=5, col=1)
    fig.update_yaxes(title_text="fraction non-zero eigenvalues", row=6, col=1)

    # Update title and height
    fig.update_layout(height=250 * 6)
    # fig.show()
    fig.write_html(cfg.Paths.plot)

    if cfg.Data.type in ['predictive', 'predictive_track']:
        fig = plot_rate_maps(cfg, np.arange(int(cfg.Data.n_samples)/2).astype(int))
        fig.savefig('rate_maps1.png', dpi=600, format='png')
        fig = plot_rate_maps(cfg, np.arange(int(cfg.Data.n_samples/2), cfg.Data.n_samples).astype(int))
        fig.savefig('rate_maps2.png', dpi=600, format='png')


def update_cfg(cfg: Scenario, update_vars):
    for var in update_vars:
        for n1 in dir(cfg):
            for n2 in dir(cfg[n1]):
                if varname([var]) == n2:
                    cfg[n1][n2] = var


def varname(var):
    return f'{var[0]=}'.partition('=')[0]


def get_tau(x_full, y_full):
    y = np.copy(y_full)
    end_val = y.max() - (y.max() - y.min()) * 0.9
    end = abs(y - end_val).argmin()
    y = y[:end]
    y -= (y.min() - 1e-10)
    x = np.copy(x_full[:end])
    p = np.polyfit(x, np.log(y), 1, w=np.sqrt(np.flip(x / x.max())))
    a = np.exp(p[1])
    b = p[0]
    return b
    # plt.plot(x_full,y_full)
    # plt.show()


def calc_rate_map(pos, hidden):
    nbins_x = 100
    pos = pos - pos.min()
    V = np.diff(pos)
    V = np.concatenate([[V[0]], V])
    pos = pos * np.sign(V)
    n_hidden = hidden.shape[-1]
    rmaps = np.zeros([n_hidden, nbins_x])
    infos = np.zeros([n_hidden])
    res = binned_statistic(pos, pos, 'count', nbins_x)
    time_mat = res[0]
    for i in range(n_hidden):
        rate_map = binned_statistic(pos, hidden[:, i], 'mean', nbins_x)[0]
        rate_map[np.isinf(rate_map)] = 0
        rate_map[np.isnan(rate_map)] = 0

        prob_mat = time_mat / np.nansum(time_mat)
        mean_rate = np.nansum(prob_mat * rate_map)
        infos[i] = np.nansum(prob_mat * (rate_map / mean_rate) * (np.log2(rate_map / mean_rate)))

        rmaps[i, :] = gaussian_filter(rate_map, sigma=[0.06 * nbins_x])
    return infos, rmaps, infos > 0.01


def calc_rate_maps(model_l, pos, model, X):
    T = len(model_l)
    infos_l = []
    rate_maps_l = []
    is_place_l = []
    for i in range(T):
        model.load_state_dict(model_l[i])
        out, hidden = model(X)
        out = out.detach().numpy();
        hidden = hidden.detach().numpy();
        infos, rate_maps, is_place = calc_rate_map(pos, hidden)
        infos_l.append(infos)
        rate_maps_l.append(rate_maps)
        is_place_l.append(is_place)
    return infos_l, rate_maps_l, is_place_l


def get_norm_rmaps(rate_maps, order):
    rmaps = rate_maps / rate_maps.max(1)[:, np.newaxis]
    rmaps[np.isnan(rmaps)] = 0
    return rmaps[order, :]


def get_order(rate_maps, is_place):
    inds = np.where(is_place)[0]
    order = rate_maps[inds].argmax(-1).argsort()
    return inds[order]


def plot_rate_maps(cfg, folder='', inds=[]):
    plt.jet()
    with open(folder + cfg.Paths.output, 'rb') as handle:
        output_dict = pkl.load(handle)[0]
    with open(folder + cfg.Paths.data, 'rb') as handle:
        data_dict = pkl.load(handle)
    t = output_dict['sample_epochs_hessian']
    T = int(len(t))
    T_print = f'$10^{int(np.ceil(np.log10(t[-1])))}$'
    X = data_dict['X'];
    pos = data_dict['pos']
    if inds != []:
        X = X[inds,:]
        pos = pos[inds]

    cfg.Network.input_dim = output_dict['model_l'][0]['hidden.weight'].shape[1]
    cfg.Network.output_dim = output_dict['model_l'][0]['output.weight'].shape[0]
    model = Net(cfg).double()

    infos_l, rate_maps_l, is_place_l = calc_rate_maps(output_dict['model_l'], pos, model, X)

    samples = np.linspace(0.1*T, (T - 1)/1, 4).astype(int)
    fig, axs = plt.subplots(4, 4, figsize=(4, 4))
    for i in range(4):
        # order = get_order(rate_maps_l[samples[i]], rate_maps_l[samples[i]].sum(1)>0)
        order = get_order(rate_maps_l[samples[i]], is_place_l[samples[i]])
        for j in range(4):
            ind = samples[j]
            rate_maps = rate_maps_l[ind]
            rate_maps = get_norm_rmaps(rate_maps, order)
            axs[i, j].imshow(rate_maps, aspect='auto', cmap=cmap, interpolation='nearest')
            axs[i, j].set_xticks([])
            if i == 0:
                axs[i, j].set_title(f"t={np.round(t[ind] / t[-1], 1)}$\cdot10^{int(np.round(np.log10(t[-1])))}$",
                                    fontsize=8)
            if j != 0:
                axs[i, j].set_yticks([])
            else:
                axs[i, j].set_yticks([0, len(order)])
    # plt.tight_layout()
    plt.show()
    return fig


def plot_rate_maps_cross_envs(cfg, folder=''):
    plt.jet()
    with open(folder + cfg.Paths.output, 'rb') as handle:
        output_dict = pkl.load(handle)[0]
    with open(folder + cfg.Paths.data, 'rb') as handle:
        data_dict = pkl.load(handle)
    t = output_dict['sample_epochs_hessian']
    T = int(len(t))
    T_print = f'$10^{int(np.ceil(np.log10(t[-1])))}$'
    X = data_dict['X'];
    pos = data_dict['pos']
    inds1 = np.arange(int(cfg.Data.n_samples)/2).astype(int)
    inds2 = np.arange(int(cfg.Data.n_samples/2), cfg.Data.n_samples).astype(int)
    X1 = X[inds1,:]
    pos1 = pos[inds1]
    X2 = X[inds2,:]
    pos2 = pos[inds2]

    cfg.Network.input_dim = output_dict['model_l'][0]['hidden.weight'].shape[1]
    cfg.Network.output_dim = output_dict['model_l'][0]['output.weight'].shape[0]
    model = Net(cfg).double()

    infos_l1, rate_maps_l1, is_place_l1 = calc_rate_maps(output_dict['model_l'], pos1, model, X1)
    infos_l2, rate_maps_l2, is_place_l2 = calc_rate_maps(output_dict['model_l'], pos2, model, X2)

    samples = np.linspace(0.1*T, (T - 1)/1, 4).astype(int)
    fig, axs = plt.subplots(4, 4, figsize=(4, 4))
    for i in range(4):
        ind = samples[i]
        order1 = get_order(rate_maps_l1[ind], is_place_l1[ind])
        order2 = get_order(rate_maps_l2[ind], is_place_l2[ind])
        order = [order1, order1, order2, order2]
        rate_maps_l = [rate_maps_l1[ind], rate_maps_l2[ind], rate_maps_l1[ind], rate_maps_l2[ind]]
        for j in range(4):
            rate_maps = rate_maps_l[j]
            rate_maps = get_norm_rmaps(rate_maps, order[j])
            axs[i, j].imshow(rate_maps, aspect='auto', cmap=cmap, interpolation='nearest')
            axs[i, j].set_xticks([])
            if j != 0:
                axs[i, j].set_yticks([])
            else:
                axs[i, j].set_yticks([0, len(order[j])])
    # plt.tight_layout()
    # plt.show()
    return fig


def cfg_to_namespace(cfg):
    for key in cfg.keys():
        cfg[key] = SimpleNamespace(**cfg[key])
    cfg = SimpleNamespace(**cfg)
    return cfg


def retrieve_data(folder):
    plt.jet()
    with open(folder + '.hydra/config.yaml') as f:
        cfg = yaml.load(f, Loader=SafeLoader)
    cfg = cfg_to_namespace(cfg)
    with open(folder + cfg.Paths.output, 'rb') as handle:
        output_dict = pkl.load(handle)[0]
    with open(folder + cfg.Paths.data, 'rb') as handle:
        data_dict = pkl.load(handle)
    d = SimpleNamespace(**output_dict)
    return cfg, output_dict, d