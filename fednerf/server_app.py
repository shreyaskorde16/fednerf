"""first: A Flower / PyTorch app."""
import pickle
from dataclasses import asdict
from typing import Iterable, Optional, Tuple
import pandas as pd
import torch
import os
import numpy as np
import logging
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.common import MetricRecord, Message
from omegaconf import OmegaConf
from fednerf.task import Net
from logging.handlers import RotatingFileHandler
from fednerf.fednerf_utils.server_utils import (
    get_config,
    custom_logging,
    get_log_dirs,
)
from fednerf.fednerf_utils.fl_run_nerf import (
    create_nerf,
)
from fednerf.my_strategy import (
    CustomFedAdagrad,
    CustomFedProx,
)



# Create ServerApp
app = ServerApp()


class CustomFedavg(FedAvg):
    def __init__(
        self,
        logger: None,
        config: dict,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.config = config

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""

        loss_avg = []
        psnr_avg = []
        psnr0_avg = []

        for reply in replies:
            if reply.has_content():
                # Retrieve the ConfigRecord from the message
                metrics = reply.content["metrics"]
                #metadata_bytes = config_record["meta"]
                # Deserialize it
                #train_meta = pickle.loads(metadata_bytes)
                #print(asdict(train_meta))
                print(type(metrics))
                print(metrics)
                loss_avg.append(metrics["train_loss"])
                psnr_avg.append(metrics["train_psnr"])
                psnr0_avg.append(metrics["train_psnr0"])
        
        if len(loss_avg) > 0:
            loss = sum(loss_avg) / len(loss_avg)
            psnr = sum(psnr_avg) / len(psnr_avg)
            psnr0 = sum(psnr0_avg) / len(psnr0_avg)
            self.logger.info(f"Server Round {server_round} - Aggregated Train Metrics: Loss: {loss:.4f}, PSNR: {psnr:.2f}, PSNR0: {psnr0:.2f}")

            avg_metrics = pd.DataFrame({
                                "round": [server_round],
                                "loss_agg": [loss],
                                "psnr_agg": [psnr],
                                "psnr0_agg": [psnr0],
                            })
            # Save to CSV
            csv_path = os.path.join(self.config["csv_dir"], "aggregated_metrics.csv")
            write_header = not os.path.exists(csv_path)
            avg_metrics.to_csv(csv_path, mode="a", header=write_header, index=False, sep=',')
            
        else:
            print(f"Server Round {server_round} - No training metrics received.")
        
        # Aggregate the ArrayRecords and MetricRecords as usual
        return super().aggregate_train(server_round, replies)
    
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
        ) -> Iterable[Message]:
        """Configure the next round of federated training and maybe do LR decay."""
        # Decrease learning rate by a factor of 0.5 every 5 rounds
        # Note: server_round starts at 1, not 0
        #if server_round % 5 == 0:
            #config["lr"] *= 0.5
            #print("LR decreased to:", config["lr"])
        # Pass the updated config and the rest of arguments to the parent class
        config["server_round"] = server_round
        return super().configure_train(server_round, arrays, config, grid)
    
    #@abstractmethod
    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of evaluation."""
        config["server_round"] = server_round
        return super().configure_evaluate(server_round, arrays, config, grid)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    config_file_name: str = context.run_config["config-file-name"]
    # config_file_name="colosseum_1.yaml"
    config = get_config(
        config_path="./configs",
        config_file_name=config_file_name,
    )

    # append log directories to config
    config = get_log_dirs(Client_id=None, cfg=config, start=True)
    # Convert the OmegaConf config to a dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Set up custom logging
    logger = custom_logging(client_id=None, cfg=config)
    #logger.info(f"Loaded config:\n{OmegaConf.to_yaml(config)}")

    nerf_model, nerf_model_fine, _, _, _, _, config_dict, config_test = create_nerf(config=config_dict)

    # Assume `model` and `model_fine` are your NeRF models
    coarse_state_dict = nerf_model.state_dict()
    fine_state_dict = nerf_model_fine.state_dict()

    # Prefix fine model keys and combine
    fine_state_dict = {f"fine_{k}": v for k, v in fine_state_dict.items()}

    combined_state_dict = {**coarse_state_dict, **fine_state_dict}

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    #num_rounds: int = context.run_config["num-server-rounds"]
    num_rounds = config_dict["global_rounds"] 

    logger.info("Loading global model...")

    #combined_state_dict = {**global_model_state_dict, **fine_state_dict}
    arrays = ArrayRecord(combined_state_dict)

    # Initialize strategy
    strategy = CustomFedavg(logger=logger, config=config_dict, fraction_train=fraction_train)
    #strategy = CustomFedAdagrad(logger=logger, config=config_dict, fraction_train=fraction_train)
    #strategy = CustomFedProx(logger=logger, config=config_dict, proximal_mu=config_dict["mu"], fraction_train=fraction_train)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(config_dict),
        evaluate_config=ConfigRecord(config_dict),
        num_rounds=num_rounds,
    )


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
    coarse_path = os.path.join(config_dict["model_dir"], "nerf_coarse.pt")
    fine_path = os.path.join(config_dict["model_dir"], "nerf_fine.pt")
    torch.save(aggregated_coarse_state_dict, coarse_path)
    torch.save(aggregated_fine_state_dict, fine_path)
    
