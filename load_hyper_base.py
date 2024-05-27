import os
import cv2
import imageio
import numpy as np
import json
from utils.flow_utils import resize_flow
from run_nerf_helpers_5 import get_grid

import glob


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    with open(os.path.join(basedir, f'dataset.json'), 'r') as f:
        dataset_info = json.load(f)
    train_ids = dataset_info['train_ids'][:10]
    val_ids = dataset_info['val_ids'][:10]

    with open(os.path.join(basedir, f'scene.json'), 'r') as f:
        json_scene = json.load(f)
    bds = np.array([json_scene['near'],json_scene['far']])
    scale = json_scene['scale']
    center = json_scene['center']

    with open(os.path.join(basedir, 'metadata.json'), 'r') as f:
        json_meta = json.load(f)

    imgs = []
    poses = []
    masks = []
    for idx in train_ids + val_ids:
        with open(os.path.join(basedir, 'camera', idx+'.json'), 'r') as f:
            camera = json.load(f)
        sw, sh =  camera['image_size']
        pose = np.eye(3, 5) 
        pose[:3, :3] = np.array(camera['orientation']).T # it works...
        #pose[:3, 3] = (np.array(cam['position']) - center) * scale * 4
        pose[:3, 3] = np.array(camera['position'])

        # CHECK: simply assume all intrinsic are same ?
        cx, cy = camera['principal_point']
        fl = camera['focal_length']
        pose[:3, 4] = np.array([sh / factor, sw / factor, fl])

        poses.append(pose)

        img = imageio.imread(os.path.join(basedir, 'rgb', '{}x'.format(factor),idx+'.png'))
        mask = imageio.imread(os.path.join(basedir, 'motion_masks',idx+'.png'))/255.
        assert img.shape[0] == sh / factor
        assert img.shape[1] == sw / factor
        imgs.append(img/255.)
        masks.append(mask/255.)



    imgs = np.stack(imgs, -1)
    poses = np.stack(poses, -1)
    masks = np.stack(masks, -1)

    # poses[2, 4, :] = poses[2, 4, :] * 1./factor
    masks = np.float32(masks > 1e-3)

    cx = cx / factor
    cy = cy / factor
    fl = fl / factor
    print(fl)


 
    

    return poses, bds, imgs, masks


def load_hyper_data(args, basedir,
                   factor=2,
                   recenter=True, bd_factor=.75,
                   spherify=False, path_zflat=False,
                   frame2dolly=10):

    poses, bds, imgs, masks = _load_data(basedir, factor=factor) # factor=2 downsamples original imgs by 2x

    print('Loaded', basedir, bds.min(), bds.max())


    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :],
                           -poses[:, 0:1, :],
                            poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(imgs, -1, 0).astype(np.float32)
    masks = np.moveaxis(masks, -1, 0).astype(np.float32)
    bds = np.repeat(np.expand_dims(bds,0),images.shape[0],0)
    # bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    # Rescale if bd_factor is provided
    # sc = 1. if bd_factor is None else 1./(np.percentile(bds[:, 0], 5) * bd_factor)

    # poses[:, :3, 3] *= sc
    # bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    # # Only for rendering
    # if frame2dolly == -1:
    #     c2w = poses_avg(poses)
    # else:
    #     c2w = poses[frame2dolly, :, :]
    H, W, _ = poses[0,:, -1]

    grids = np.repeat(np.empty((1, int(H), int(W), 8), np.float32),images.shape[0],axis=0)

    # grids = get_grid(int(H), int(W), len(poses), flows_f, flow_masks_f, flows_b, flow_masks_b) # [N, H, W, 8]

    return images, masks, poses, bds,grids



def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

def nearest_pose(p, poses):
    dists = np.sum(np.square(p[:3, 3:4] - poses[:3, 3, :]), 0)
    return np.argsort(dists)

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w



