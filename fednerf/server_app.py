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
    CustomFedAvg,
)
from fednerf.fednerf_utils.fl_run_nerf import (
    create_nerf,
)


# Create ServerApp
app = ServerApp()


def get_config_fn(config_dict):
    """Return a function that returns a ConfigRecord containing the config."""
    def config_fn(rnd: int):
        return ConfigRecord(config_dict)
    return config_fn


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    config = get_config(
        config_path="./configs",
        config_file_name="colosseum_1.yaml",
    )

    # append log directories to config
    config = get_log_dirs(Client_id=None, cfg=config)
    # Convert the OmegaConf config to a dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    
    # Set up custom logging
    logger = custom_logging(client_id=None, cfg=config)
    logger.info(f"Loaded config:\n{OmegaConf.to_yaml(config)}")

    nerf_model, nerf_model_fine, _, _, _, _, config_dict, config_test = create_nerf(config=config_dict)

    # Assume `model` and `model_fine` are your NeRF models
    coarse_state_dict = nerf_model.state_dict()
    fine_state_dict = nerf_model_fine.state_dict()

    # Print coarse model state dict
    print("\nCoarse Model State Dictionary:")
    for key, tensor in coarse_state_dict.items():
        print(f"{key}: {tensor.shape}")
    print("*"*50)

    # Print fine model state dict
    print("\nFine Model State Dictionary:")
    for key, tensor in fine_state_dict.items():
        print(f"{key}: {tensor.shape}")
    print("*"*50)

    # Prefix fine model keys and combine
    fine_state_dict = {f"fine_{k}": v for k, v in fine_state_dict.items()}
    # Print fine model state dict
    print("\nModified Fine Model State Dictionary:")
    for key, tensor in fine_state_dict.items():
        print(f"{key}: {tensor.shape}")
    print("*"*50)

    combined_state_dict = {**coarse_state_dict, **fine_state_dict}

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]

    logger.info("Loading global model...")

    #combined_state_dict = {**global_model_state_dict, **fine_state_dict}
    arrays = ArrayRecord(combined_state_dict)

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train)
                      #on_evaluate_config_fn=get_config_fn(config_dict))
    #strategy = CustomFedAvg(config=config_dict, fraction_fit=fraction_train)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(config_dict),
        evaluate_config=ConfigRecord(config_dict),
        num_rounds=num_rounds,
   

    )

    """
    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

    # Save final models to disk
    print("\nSaving final models to disk...")
    aggregated_combined_state_dict = result.arrays.to_torch_state_dict()

    # Split the aggregated state dictionary
    aggregated_coarse_state_dict = {}
    aggregated_fine_state_dict = {}

    for key, value in aggregated_combined_state_dict.items():
        if key.startswith("fine_"):
            aggregated_fine_state_dict[key[len("fine_"):]] = value
        else:
            aggregated_coarse_state_dict[key] = value

    # Save the aggregated models
    torch.save(aggregated_coarse_state_dict, "final_nerf_coarse.pt")
    torch.save(aggregated_fine_state_dict, "final_nerf_fine.pt")
    """
