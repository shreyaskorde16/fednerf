"""Custom strategy for FedNeRF training."""

import os
import pandas as pd
import torch
import os
import numpy as np
from omegaconf import DictConfig, OmegaConf
from flwr.serverapp.strategy import FedAvg, FedAdagrad
from flwr.common import (
    Parameters,
    ConfigRecord,
    FitRes,
    EvaluateRes,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    MetricRecord, 
    Message
)
import pickle
from flwr.server.client_manager import ClientManager
from typing import Dict
from dataclasses import asdict
from typing import Iterable, Optional
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp




class CustomFedAdagrad(FedAdagrad):
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
        print("In Custom Fedadagrad")

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

