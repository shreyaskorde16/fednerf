"""first: A Flower / PyTorch app."""

import torch
import logging
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from omegaconf import OmegaConf
from fednerf.task import Net
from logging.handlers import RotatingFileHandler
from fednerf.fednerf_utils.server_utils import (
    get_config,
    custom_logging,
    get_log_dirs,
)

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    config = get_config(
        config_path="./configs",
        config_file_name="colosseum_1.yaml",
    )

    # append log directories to config
    config = get_log_dirs(Client_id=1, cfg=config)

    # Set up custom logging
    logger = custom_logging(client_id=None, cfg=config)
    logger.info(f"Loaded config:\n{OmegaConf.to_yaml(config)}")

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Load global model
    global_model = Net()
    logger.info("Loading global model...")
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
