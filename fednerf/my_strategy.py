"""Custom strategy for FedNeRF training."""

import os
from omegaconf import DictConfig, OmegaConf
from flwr.serverapp.strategy import FedAvg
from flwr.common import (
    Parameters,
    ConfigRecord,
    FitRes,
    EvaluateRes,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from typing import Dict



