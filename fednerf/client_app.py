"""first: A Flower / PyTorch app."""

import torch
import os
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from omegaconf import OmegaConf
from fednerf.task import Net, load_data
from fednerf.task import test as test_fn
from fednerf.task import train as train_fn
from fednerf.fednerf_utils.server_utils import (
    custom_logging,
)
from fednerf.fednerf_utils.load_llff import load_llff_data
from fednerf.fednerf_utils.load_blender import load_blender_data
from fednerf.fednerf_utils.load_LINEMOD import load_LINEMOD_data
from fednerf.fednerf_utils.load_deepvoxels import load_dv_data

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    config = msg.content["config"]
    print(config["lr"])
    print(config["expname"])
    config["client_id"] = context.node_config["partition-id"]
    print(config["client_id"])
    cid = context.node_config["partition-id"]

    # Custom logging
    logger = custom_logging(client_id=cid, cfg=config)
    logger.info(f"Loaded config:\n{config}")
    cid_datadir = os.path.join(config["datadir"], f"colosseum_{cid}_processed")
    #print(f"Client {cid} using data from {cid_datadir}")

    K = None
    if config["dataset_type"] == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(cid_datadir, config["factor"],
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=config["spherify"])
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        logger.info(f"Loaded llff, image_shape = {images.shape}, render_poses_shape = {render_poses.shape}, hwf = {hwf}, Client data_path = {cid_datadir}")
        logger.info(f"render_poses_shape = {render_poses.shape}, hwf = {hwf}")
        logger.info(f"Client data_path = {cid_datadir}")
        if not isinstance(i_test, list):
            i_test = [i_test]

        if config["llffhold"] > 0:
            logger.info(f"Auto LLFF holdout = {config['llffhold']}")
            i_test = np.arange(images.shape[0])[::config["llffhold"]]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        logger.info('DEFINING BOUNDS')
        if config["no_ndc"]:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        logger.info(f"NEAR = {near}, FAR = {far}")
    
    elif config["dataset_type"] == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(cid_datadir, config["half_res"], config["testskip"])
        print('Loaded blender', images.shape, render_poses.shape, hwf, cid_datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if config["white_bkgd"]:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
    
    elif config["dataset_type"] == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(cid_datadir, config["half_res"], config["testskip"])
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if config["white_bkgd"]:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
    
    elif config["dataset_type"] == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=config["shape"],
                                                                 basedir=cid_datadir,
                                                                 testskip=config["testskip"])

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, cid_datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', config["dataset_type"], 'exiting')
        return
    
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


    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
