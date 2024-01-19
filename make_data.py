import copy

from torch.optim import SGD, Adam
from model import *
import numpy as np
from config import Scenario
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import interp1d


def make_data(cfg: Scenario):
    n_samples = cfg.Data.n_samples
    X_dim = cfg.Data.X_dim
    y_dim = cfg.Data.y_dim
    type = cfg.Data.type
    sigma = cfg.Data.smooth_factor
    data = {}
    if cfg.Data.type == 'random':
        X = torch.tensor(np.random.random([n_samples, X_dim])).double()
        y = torch.tensor(np.random.random([n_samples, y_dim])).double()

    elif type == 'smoothed':
        X = torch.tensor(gaussian_filter(np.random.random([n_samples, X_dim]), sigma=[n_samples/sigma, y_dim/5])).double()
        y = torch.tensor(gaussian_filter(np.random.random([n_samples, y_dim]), sigma=[n_samples/sigma, y_dim/5])).double()

    if cfg.Data.type == 'random_autoencoder':
        X = torch.tensor(np.random.random([n_samples, X_dim])).double()
        y = copy.deepcopy(X)
        cfg.Data.y_dim = cfg.Data.X_dim

    elif type == 'smoothed_autoencoder':
        X = torch.tensor(gaussian_filter(np.random.random([n_samples, X_dim]), sigma=[n_samples/sigma, y_dim/5])).double()
        y = copy.deepcopy(X)
        cfg.Data.y_dim = cfg.Data.X_dim

    elif type == 'predictive':
        cfg.Data.y_dim = cfg.Data.X_dim
        arena_size = 120
        FOV = 30
        step_size = arena_size / n_samples * 20
        visual_scene = gaussian_filter(torch.tensor(np.random.random(arena_size+1)).double(), sigma=arena_size/FOV)
        visual_scene = min_max_norm(visual_scene)
        visual_scene = interp1d(np.arange(arena_size+1), visual_scene)
        pos = np.cumsum(np.random.uniform(size=n_samples+1)*step_size)
        data['pos'] = pos[:-1] % arena_size
        pos_v = (pos[:, np.newaxis] + np.linspace(0, FOV, X_dim)[np.newaxis, :]) % arena_size
        X = torch.tensor(visual_scene(pos_v))
        y = X[1:]
        X = X[:-1]

    elif type == 'predictive_track':
        cfg.Data.y_dim = cfg.Data.X_dim
        X, y, pos, theta = run_traj(cfg)
        data['pos'] = pos
        data['theta'] = theta
        X[:,-1] = min_max_norm(X[:,-1])
        X[:,:-1] = min_max_norm(X[:,:-1])
        y = min_max_norm(y)
        cfg.Data.X_dim = cfg.Data.X_dim + 1
        # cfg.Data.n_samples *= cfg.Env.num_envs

    X = min_max_norm(X)
    y = min_max_norm(y)
    data['X'] = X
    data['y'] = y
    with open(cfg.Paths.data, 'wb') as handle:
        pkl.dump(data, handle)


def min_max_norm(x):
    x -= x.min()+1e-5
    x /= x.max()+1e-5
    return (x - 0.5)*2


def run_traj(cfg:Scenario, cont=False):
    data = {}

    trial_len = int(cfg.Data.n_samples/cfg.Env.num_envs)
    num_envs = cfg.Env.num_envs
    FOV_tiles = cfg.Data.y_dim
    FOV_angle = np.pi / 4
    len_action = 1
    env_type = 'Env_1d'

    data['pos_x'] = np.zeros([trial_len*num_envs])
    data['pos'] = np.zeros([trial_len*num_envs])
    data['theta'] = np.zeros([trial_len*num_envs])
    data['theta_v'] = np.zeros([trial_len*num_envs])
    data['X'] = torch.Tensor(trial_len*num_envs, FOV_tiles+len_action).double()
    data['y'] = torch.Tensor(trial_len*num_envs, FOV_tiles).double()

    for env_num in range(cfg.Env.num_envs):
        env = Env_2d(cfg)
        agent = Agent()
        env.add_agent(agent)

        with torch.no_grad():
            for t in range(env_num*trial_len, (env_num+1)*trial_len):
                obs = torch.tensor(agent.visual_scene).double()
                action = torch.tensor(agent.theta_V).unsqueeze(0).double()

                data['pos_x'][t] = agent.x
                data['pos'][t] = agent.y
                data['theta'][t] = agent.theta
                data['theta_v'][t] = agent.theta_V
                data['X'][t, :] = torch.cat([obs, action])
                env.run_step(cont)
                data['X'][t, -1] = agent.theta - data['theta'][t]
                data['y'][t, :] = torch.tensor(agent.visual_scene).double()

        # data['X'] = torch.tensor(data['X']); data['y'] = torch.tensor(data['y'])
    return data['X'], data['y'], data['pos'], data['theta']


class Env_2d:
    def __init__(self, cfg:Scenario):
        self.size_x = cfg.Env.size_x
        self.size_y = cfg.Env.size_y
        self.kernel = cfg.Env.kernel
        self.FOV_tiles = cfg.Data.y_dim
        self.theta_v_sigma = np.pi * 0.1 / 360
        self.FOV_angle = np.pi/cfg.Env.FOV_angle_frac
        self.V0 = cfg.Env.V0

        self.buffer_x = self.size_x * 0.5
        self.buffer_y = self.size_y * 0.1
        self.walls = [self.size_y / 2, self.size_x / 2, -self.size_y / 2, -self.size_x / 2]
        self.xlims = [-self.size_x / 2 + self.buffer_x, self.size_x / 2 - self.buffer_x]
        self.ylims = [-self.size_y / 2 + self.buffer_y, self.size_y / 2 - self.buffer_y]
        self.circumference = 2 * self.size_x + 2 * self.size_y

        #         if self.size_x < 2 or self.size_y < 2:
        #             raise Exception('Arena size too small')
        min_field = 2 * self.FOV_tiles
        self.n_tiles = int((self.size_x + self.size_y) * min_field)
        kernel = self.n_tiles * self.kernel

        # leng = int(self.FOV_tiles * self.size_x)
        # x = np.arange(leng) / leng
        # triangle = ((x - 0.1) * (x > 0.1) - 2 * (x - 0.5) * (x > 0.5)) * (x < 0.9)
        # triangle /= triangle.max()
        # square = (1 * (x > 0.1)) * (x < 0.9) - triangle
        # leng = int(self.FOV_tiles * self.size_y)
        # x = np.arange(leng) / leng
        # cos = (np.cos((x - 0.1) * 2 * np.pi) * (x > 0.1)) * (x < 0.9)
        # cos /= triangle.max()
        # power = ((x - 0.1) ** 2 * (x > 0.1)) * (x < 0.9)
        # power /= triangle.max()
        # self.obs = np.concatenate([triangle, cos, square, power])

        self.obs = gaussian_filter1d(np.tile(np.random.randn(self.n_tiles), 3), kernel)[self.n_tiles:2 * self.n_tiles]
        self.obs_intep = np.concatenate([[self.obs[-1]], self.obs, [self.obs[0]]])
        self.obs_cords = np.arange(-1, self.n_tiles + 1)
        self.agents = []
        self.crosses = []

    def add_agent(self, agent):
        agent.V = self.V0
        self.agents.append(agent)
        self.agent_see(self.agents[-1])

    def run_step(self, cont):
        for agent in self.agents:
            hit_boundary = self.agent_move(agent, cont)
            self.agent_see(agent)
            agent.theta_V = np.random.normal(0, self.theta_v_sigma)  # theta_v random walk
            if any([hit_boundary[2]]):
                agent.theta = 0
            if any([hit_boundary[3]]):
                agent.theta = np.pi

    def agent_move(self, agent, cont):
        returning_force = 1 - 0.5 * np.linalg.norm([agent.x, agent.y]) / np.linalg.norm([self.xlims[0], self.ylims[0]])
        agent.V = self.V0 + np.random.normal(0,0.1)
        if cont:
            agent.x = round(agent.x + np.sin(agent.theta) * agent.V, 1)
            agent.y = round(agent.y + np.cos(agent.theta) * agent.V, 1)
        else:
            agent.x += np.sin(agent.theta) * agent.V
            agent.y += np.cos(agent.theta) * agent.V
        hit_boundary = self.check_boundary(agent)
        agent.theta = (agent.theta + agent.theta_V * 2 * np.pi) % (2 * np.pi)
        return hit_boundary

    def check_boundary(self, agent):
        hit_boundary = [False, False, False, False]
        if agent.x < self.xlims[0]:
            agent.x = self.xlims[0]
            hit_boundary[0] = True
        if agent.x > self.xlims[1]:
            agent.x = self.xlims[1]
            hit_boundary[1] = True
        if agent.y < self.ylims[0]:
            agent.y = self.ylims[0]
            hit_boundary[2] = True
        if agent.y > self.ylims[1]:
            agent.y = self.ylims[1]
            hit_boundary[3] = True
        return hit_boundary

    def agent_see(self, agent):
        agent.cross1 = self.intersect_pos(agent.theta - self.FOV_angle, agent.x, agent.y)
        agent.cross2 = self.intersect_pos(agent.theta + self.FOV_angle, agent.x, agent.y)
        agent.cross_mid = self.intersect_pos(agent.theta, agent.x, agent.y)

        agent.visual_scene = np.zeros(self.FOV_tiles)
        self.crosses = []
        for i, angle in enumerate(
                np.linspace(agent.theta - self.FOV_angle, agent.theta + self.FOV_angle, self.FOV_tiles)):
            self.crosses.append(self.intersect_pos(angle, agent.x, agent.y))
            cross = self.intersect_pos(angle, agent.x, agent.y)
            phase = self.get_phase(cross)
            # remove interp
            agent.visual_scene[i] = np.interp(phase * self.n_tiles, self.obs_cords, self.obs_intep)
        # phase2 = self.get_phase(agent.cross2)
        # agent.visual_scene = self.get_visual_scene(phase1, phase2)

    def get_phase(self, cross):
        phase = cross[2] / 4
        if cross[2] == 0: phase += (cross[0] - self.walls[3]) / self.circumference
        if cross[2] == 1: phase += (self.walls[0] - cross[1]) / self.circumference
        if cross[2] == 2: phase += (self.walls[1] - cross[0]) / self.circumference
        if cross[2] == 3: phase += (cross[1] - self.walls[2]) / self.circumference
        return phase

    def get_visual_scene(self, phase1, phase2):
        ind1, ind2 = [int(self.n_tiles * phase1), int(self.n_tiles * phase2)]
        if phase1 > phase2:
            left_over = (1 - phase1) / (1 - phase1 + phase2)
            FOV1 = int(left_over * self.FOV_tiles)
            inds1 = np.linspace(ind1, self.n_tiles, FOV1) - 1
            visual_scene = self.obs[np.linspace(0, ind2, self.FOV_tiles - FOV1).astype(int)]
            if len(inds1):
                visual_scene = np.concatenate([self.obs[inds1.astype(int)], visual_scene])
        else:
            visual_scene = self.obs[np.linspace(ind1, ind2 - 1, self.FOV_tiles).astype(int)]
        return visual_scene

    def intersect_pos(self, theta, x0, y0):
        theta = theta % (2 * np.pi)
        crosses = []
        for i in range(4):
            if i * np.pi / 2 <= theta < (i + 1) * np.pi / 2:
                if not i % 2:
                    wall_0_cross = (self.walls[i] - y0) * np.tan(theta) + x0
                    wall_1_cross = (self.walls[(i + 1) % 4] - x0) / np.tan(theta) + y0
                    wall_1_crossed = abs(wall_1_cross) <= abs(self.walls[i])
                    cross = [self.walls[i + 1], wall_1_cross, i + 1] if wall_1_crossed else [wall_0_cross,
                                                                                             self.walls[i], i]
                else:
                    wall_0_cross = (self.walls[i] - x0) / np.tan(theta) + y0
                    wall_1_cross = (self.walls[(i + 1) % 4] - y0) * np.tan(theta) + x0
                    wall_1_crossed = abs(wall_1_cross) <= abs(self.walls[i])
                    cross = [wall_1_cross, self.walls[(i + 1) % 4], (i + 1) % 4] if wall_1_crossed else [self.walls[i],
                                                                                                         wall_0_cross,
                                                                                                         i]
        return cross

class Agent:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.V = 0
        self.theta = (np.random.uniform() - 1) / 100
        self.theta_V = 0
        self.cross1 = 0
        self.cross2 = 0
        self.cross_mid = 0
        self.visual_scene = 0