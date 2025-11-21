"""Client specific helper functions for FedNeRF"""
import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from fednerf.fednerf_utils.fl_run_nerf import (
    raw2outputs,
    get_rays_np,
    render,
    render_path
)
from fednerf.fednerf_utils.fl_run_nerf_helpers import (
    get_rays,
)
from fednerf.fednerf_utils.load_llff import load_llff_data
from fednerf.fednerf_utils.load_blender import load_blender_data
from fednerf.fednerf_utils.load_LINEMOD import load_LINEMOD_data
from fednerf.fednerf_utils.load_deepvoxels import load_dv_data



def train_fednerf(H, W, K, poses, i_train, i_test, i_val, start, 
                  nerf_model, nerf_model_fine, network_query_fn, 
                  render_poses, device, optimizer, 
                  hwf, images, logger,server_round, config):
    """Train the FedNeRF model on local data."""

    cid = config['client_id']
    logger.info(f"Client {cid} training started.")
    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    # Misc
    img2mse = lambda x, y : torch.mean((x - y) ** 2)
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))
    to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
    
    # Prepare raybatch tensor if batching random rays
    N_rand = config["N_rand"]
    use_batching = not config["no_batching"]
    if use_batching:
        # For random ray batching
        #print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        #print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        #print('shuffle rays')
        np.random.shuffle(rays_rgb)

        #print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_local_iters = config["local_iterations"] + 1
    #print('Begin')
    logger.info(f'TRAIN views for Client Id: {cid} --> {len(i_train)}')
    logger.info(f'TEST views for Client Id: {cid} --> {len(i_test)}')
    logger.info(f'VAL views for Client Id: {cid} --> {len(i_val)}')

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    global_step = 0
    start = global_step + 1
    for i in trange(start, N_local_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                # shuffle data after epochs
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < config["precrop_iters"]:
                    dH = int(H//2 * config["precrop_frac"])
                    dW = int(W//2 * config["precrop_frac"])
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        logger.info(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {config['precrop_iters']}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=config["chunk"], rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                ndc=config["ndc"],
                                                near=config["near"], far=config["far"],
                                                use_viewdirs=config["use_viewdirs"],
                                                model=nerf_model,
                                                fine_model=nerf_model_fine,
                                                nerf_query_fn=network_query_fn,
                                                config=config,
                                                device=device)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = config["lrate_decay"] * 1000
        new_lrate = config["lrate"] * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####
        config_test = copy.deepcopy(config)
        config_test["perturb"] = False
        config_test['raw_noise_std'] = 0.

        if i%config["i_print"]==0 or i < 10:

            tqdm.write(f"************ [TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} Client_Id: {cid} Server Round: {server_round} ************ ")

            logger.info(
                f"************ [TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} Client_Id: {cid} Server Round: {server_round} ************ \n"
            )

        global_step += 1

    return nerf_model, nerf_model_fine, loss.item(), psnr.item(), psnr0.item()




def load_nerf_data(config = None, cid_datadir = None, logger = None):
    
    K = None
    if config["dataset_type"] == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(cid_datadir, config["factor"],
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=config["spherify"])
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        logger.info(f"Loadeed {config['dataset_type']} data for client ID {config['client_id']}")
        #logger.info(f"Loaded llff, image_shape = {images.shape}, render_poses_shape = {render_poses.shape}, hwf = {hwf}, Client data_path = {cid_datadir}")
        #logger.info(f"render_poses_shape = {render_poses.shape}, hwf = {hwf}")
        #logger.info(f"Client data_path = {cid_datadir}")
        if not isinstance(i_test, list):
            i_test = [i_test]

        if config["llffhold"] > 0:
            #logger.info(f"Auto LLFF holdout = {config['llffhold']}")
            i_test = np.arange(images.shape[0])[::config["llffhold"]]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        #logger.info('DEFINING BOUNDS')
        if config["no_ndc"]:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        #logger.info(f"NEAR = {near}, FAR = {far}")
    
    elif config["dataset_type"] == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(cid_datadir, config["half_res"], config["testskip"])
        logger.info(f"Loadeed {config['dataset_type']} data for client ID {config['client_id']}")
        #print('Loaded blender', images.shape, render_poses.shape, hwf, cid_datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if config["white_bkgd"]:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
    
    elif config["dataset_type"] == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(cid_datadir, config["half_res"], config["testskip"])
        logger.info(f"Loadeed {config['dataset_type']} data for client ID {config['client_id']}")
        #print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        #print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if config["white_bkgd"]:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
    
    elif config["dataset_type"] == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=config["shape"],
                                                                 basedir=cid_datadir,
                                                                 testskip=config["testskip"])

        #print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, cid_datadir)
        logger.info(f"Loadeed {config['dataset_type']} data for client ID {config['client_id']}")
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', config["dataset_type"], 'exiting')
        return
    
    #near = float(near)
    #far = float(far)
    
    return images, poses, render_poses, hwf, K, near, far, i_train, i_val, i_test
    