"""Server specific helper functions for FedNeRF"""

import os
import logging
from datetime import datetime
from omegaconf import OmegaConf, DictConfig
from logging.handlers import RotatingFileHandler




def get_config(config_path: str = "./configs", config_file_name: str = "colosseum_1.yaml"):
    """Load the config file"""

    config = OmegaConf.load(os.path.join(config_path, config_file_name))

    # Allow dynamic keys
    OmegaConf.set_struct(config, False)

    return config

def custom_logging(client_id: None, cfg = None):
    """Set up custom logging for Flower server and Ray."""
    # Create a single formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    root_log_path = cfg["root_log_path"]
    if client_id is None:
        log_filename = "server.log"
    elif client_id is not None:
        log_filename = f"client_{client_id}.log"
    log_filename = os.path.join(root_log_path, log_filename)

    # handlers (console and file)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=1024 * 1024,
        backupCount=5,
        mode="a",
    )
    file_handler.setFormatter(formatter)

    # Configure the root logger with both handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [console_handler, file_handler]  # Replace any existing handlers

    # Configure Flower and Ray loggers to propagate to the root logger
    flwr_logger = logging.getLogger("flwr")
    flwr_logger.setLevel(logging.INFO)
    flwr_logger.propagate = True  # Allow messages to propagate to the root logger
    flwr_logger.handlers = []     # Remove any existing handlers

    ray_logger = logging.getLogger("ray")
    ray_logger.setLevel(logging.INFO)
    ray_logger.propagate = True   # Allow messages to propagate to the root logger
    ray_logger.handlers = []      # Remove any existing handlers

    # Get a logger for your app
    logger = logging.getLogger(__name__)
    return logger

def get_log_dirs(Client_id = None, cfg: DictConfig = None):
    """ Create log directories for server and clients """
    date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    plt_dir = None
    model_dir = None
    csv_dir = None
    exp_name = cfg.expname

    if Client_id is not None:
        Cl_id = f"Client_{Client_id}"
    elif Client_id is None:
        Cl_id = "Server"

    root_log_path = os.path.join("./logs", f"Experiment_{Cl_id}_{exp_name}_{date}")
    os.makedirs(root_log_path, exist_ok=True)

    json_path = os.path.join(root_log_path, "config.json")
    OmegaConf.save(config= cfg, f= json_path)

    plt_dir = os.path.join(root_log_path, f"Exp_{Cl_id}_{exp_name}_plots")
    model_dir = os.path.join(root_log_path, f"Exp_{Cl_id}_{exp_name}_models")
    csv_dir = os.path.join(root_log_path, f"Exp_{Cl_id}_{exp_name}_csvs")
    os.makedirs(plt_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)       
    os.makedirs(csv_dir, exist_ok=True)

    cfg.root_log_path = root_log_path
    cfg.plt_dir = plt_dir    
    cfg.model_dir = model_dir
    cfg.csv_dir = csv_dir

    return cfg





