from dataclasses import dataclass
from typing import Literal


_DATATYPE = Literal["random", "smoothed", "random_autoencoder", "smoothed_autoencoder", "predictive", "predictive_track"]
_ALGOTYPE = Literal["SGD", "Adam", "RLgrad"]

@dataclass
class hydra:
    @dataclass
    class run:
        dir: str

@dataclass
class Paths:
    log: str
    data: str
    model: str
    output: str
    plot: str


@dataclass
class Network:
    n_hid: int
    regime: str
    input_dim: int
    output_dim: int


@dataclass
class Analyzer:
    num_epoch: int
    num_sample: int
    num_sample_hess: int
    stop_after: int
    converge_sample: bool


@dataclass
class Trainer:
    seed_fixed: bool
    stop_loss: bool
    lr: float
    batch_size: int
    Algorithm: _ALGOTYPE
    input_noise: float
    label_noise: float
    update_noise: float


@dataclass
class Data:
    type: _DATATYPE
    n_samples: int
    X_dim: int
    y_dim: int
    smooth_factor: int

@dataclass
class Env:
    size_x: int
    size_y: int
    kernel: float
    FOV_angle_frac: int
    V0: float
    num_envs: int = 1

@dataclass
class Scenario:
    Analyzer: Analyzer
    Trainer: Trainer
    Network: Network
    Paths: Paths
    Data: Data
    Env: Env

