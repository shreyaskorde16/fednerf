"""first: A Flower / PyTorch app."""

import torch
import os
import numpy as np
import copy
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from omegaconf import OmegaConf
from fednerf.task import Net, load_data
from fednerf.task import test as test_fn
from fednerf.task import train as train_fn
from fednerf.fednerf_utils.server_utils import (
    custom_logging,
)
from fednerf.fednerf_utils.fl_run_nerf import (
    create_nerf,
    render_path
)
from fednerf.fednerf_utils.client_utils import (
    train_fednerf,
    load_nerf_data
)

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load config from server message
    config = msg.content["config"]

    config["client_id"] = context.node_config["partition-id"]
    cid = context.node_config["partition-id"]

    # Custom logging
    logger = custom_logging(client_id=cid, cfg=config)

    #logger.info(f"Loaded config:\n{config}")
    cid_datadir = os.path.join(config["datadir"], f"colosseum_{cid}_processed")

    # Load NeRF data
    images, poses, render_poses, hwf, K, near, far, i_train, i_val, i_test = load_nerf_data(config = config,
                                                                                                cid_datadir=cid_datadir,
                                                                                                logger=logger)
    logger.info(f"Length of training data for client {cid}: {len(i_train)}")
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if config["render_test"]:
        render_poses = np.array(poses[i_test])

    # Load NeRF Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(0)

    # import model and assign start value
    #global_step = start 
    config['near'] = near
    config['far'] = far

    #model, model_fine, network_query_fn, start, optimizer, grad_vars, config, config_test = create_nerf(config=config)

    (
    nerf_model,            # Main NeRF model
    nerf_model_fine,       # Fine model
    network_query_fn,      # Query function for the network
    start,                 # Starting value
    optimizer,             # Optimizer
    grad_vars,             # Gradient variables
    config,                # Configuration (possibly updated)
    config_test            # Test configuration
    ) = create_nerf(config=config)

    # Load the combined state dictionary from the server
    combined_state_dict = msg.content["arrays"].to_torch_state_dict()

    # Split the combined state dictionary into coarse and fine models
    coarse_state_dict = {}
    fine_state_dict = {}
    # Load the model and initialize it with the received weights
    #model = Net()

    for key, value in combined_state_dict.items():
        if key.startswith("fine_"):
            fine_state_dict[key[len("fine_"):]] = value
        else:
            coarse_state_dict[key] = value

        # Load the state dictionaries into the models
    nerf_model.load_state_dict(coarse_state_dict)
    if nerf_model_fine:
        nerf_model_fine.load_state_dict(fine_state_dict)

    nerf_model.to(device)
    if nerf_model_fine:
        nerf_model_fine.to(device)

    # Train nerf model on local data
    nerf_model, nerf_model_fine, loss, psnr, psnr0 = train_fednerf(H, W, K, poses, i_train, i_test, i_val, start, 
                                                    nerf_model, nerf_model_fine, network_query_fn, 
                                                    render_poses, device, optimizer, 
                                                    hwf, images, logger, config)





    coarse_state_dict = nerf_model.state_dict()
    fine_state_dict = nerf_model_fine.state_dict()

    # Prefix the fine model's keys and combine the state dictionaries
    fine_state_dict = {f"fine_{k}": v for k, v in fine_state_dict.items()}
    combined_state_dict = {**coarse_state_dict, **fine_state_dict}

    # Construct and return reply Message
    model_record = ArrayRecord(combined_state_dict)
    metrics = {
        "train_loss": loss,
        "train_psnr": psnr,
        "train_psnr0": psnr0,
        "num-examples": len(i_train),
        "client_id": cid,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)





@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    config = msg.content["config"]
    config["client_id"] = context.node_config["partition-id"]

    # load test data
    cid = context.node_config["partition-id"]

    if "root_log_path" not in config:
        root_log_path = os.path.join(config["basedir"], config["expname"])
        os.makedirs(root_log_path, exist_ok=True)
        config["root_log_path"] = root_log_path

    # Custom logging
    logger = custom_logging(client_id=cid, cfg=config)
    #logger.info(f"Loaded config:\n{config}")
    cid_datadir = os.path.join(config["datadir"], f"colosseum_{cid}_processed")
    #print(f"Client {cid} using data from {cid_datadir}")

    # Load NeRF data
    images, poses, render_poses, hwf, K, near, far, i_train, i_val, i_test = load_nerf_data(config = config,
                                                                                                cid_datadir=cid_datadir,
                                                                                                logger=logger)

    (
    nerf_model,            # Main NeRF model
    nerf_model_fine,       # Fine model
    network_query_fn,      # Query function for the network
    start,                 # Starting value
    optimizer,             # Optimizer
    grad_vars,             # Gradient variables
    config,                # Configuration (possibly updated)
    config_test            # Test configuration
    ) = create_nerf(config=config)


    combined_state_dict = msg.content["arrays"].to_torch_state_dict()

    # Split the combined state dictionary into coarse and fine models
    coarse_state_dict = {}
    fine_state_dict = {}
    for key, value in combined_state_dict.items():
        if key.startswith("fine_"):
            fine_state_dict[key[len("fine_"):]] = value
        else:
            coarse_state_dict[key] = value

    # Load the state dictionaries into the models
    nerf_model.load_state_dict(coarse_state_dict)
    nerf_model_fine.load_state_dict(fine_state_dict)

    #model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nerf_model.to(device)
    nerf_model_fine.to(device)

    config_test = copy.deepcopy(config)
    config_test["perturb"] = False
    config_test['raw_noise_std'] = 0.

    root_log_path = config["root_log_path"]
    expname = config["expname"]

    # Evaluate the model on local test data

    #if i%config["i_testset"]==0 and i > 0:
    testsavedir = os.path.join(root_log_path, expname, 'client_{}'.format(cid),'testset_')
    os.makedirs(testsavedir, exist_ok=True)
    #logger.info('test poses shape', poses[i_test].shape)
    """
    with torch.no_grad():
        render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, config["chunk"], 
                    config_test, 
                    gt_imgs=images[i_test], 
                    savedir=testsavedir, 
                    model=nerf_model,
                    model_fine=nerf_model_fine, 
                    nerf_query_fn=network_query_fn)
    print('Saved test set')
    """
    success_message = f"Client {cid} evaluation completed successfully."
    logger.info(success_message)
    

    # Construct and return reply Message
    metrics = {
        "num-examples": len(i_test),

    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)
