import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from jaxtyping import Bool, Float, Int64
from einops import einsum, rearrange, reduce, repeat
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)

def check_rot(rot: np.ndarray, right_handed=True, eps=1e-6):
    assert np.allclose(rot.transpose() @ rot, np.eye(3), atol=eps)
    assert np.linalg.det(rot) - 1 < eps * 2
    
    if right_handed:
        assert np.abs(np.dot(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2]) - 1.0) < 1e-3
    else:
        assert np.abs(np.dot(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2]) + 1.0) < 1e-3


def plot_cooridinate_c2w(poses: np.ndarray, right_handed=True, plot_kf=False, title=None, text="", plot_every_nth_pose=1, ax=None):
    if poses.ndim == 2:
        poses = poses[None, :, :]
    assert poses.ndim == 3
    
    for i in range(poses.shape[0]):
        rot = poses[i, :3, :3]
        check_rot(rot, right_handed=right_handed)
    
        ####### Plotting #######
    new_plot = False
    if ax is None:
        new_plot = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    if plot_kf:
        color = ["r", "g", "b"]
        label = ["x kf", "y kf", "z kf"]
    else:
        color = ["c", "m", "y"]
        label = ["x all", "y all", "z all"]
    num_poses = poses.shape[0]

    # Positions
    pos = np.zeros((num_poses, 3))
    pos[:, 0] = np.asarray(poses[:, 0, 3])
    pos[:, 1] = np.asarray(poses[:, 1, 3])
    pos[:, 2] = np.asarray(poses[:, 2, 3])

    # Orientations
    # orientation is given as transposed rotation matrices
    x_ori = []
    y_ori = []
    z_ori = []
    for k in range(num_poses):
        x_ori.append(poses[k, :, 0])
        y_ori.append(poses[k, :, 1])
        z_ori.append(poses[k, :, 2])

    x_mod = 1
    y_mod = 1
    z_mod = 1
    
    # multiply by 10 to visualize orientation more clearly
    pos = pos * [x_mod, y_mod, z_mod]
    dir_vec_x = pos[0] + x_mod * x_ori[0]
    dir_vec_y = pos[0] + y_mod * y_ori[0]
    dir_vec_z = pos[0] + z_mod * z_ori[0]

    if title is not None:
        ax.plot(
            [pos[0, 0], dir_vec_x[0]],
            [pos[0, 1], dir_vec_x[1]],
            [pos[0, 2], dir_vec_x[2]],
            color=color[0],
            label=label[0],
        )
        ax.plot(
            [pos[0, 0], dir_vec_y[0]],
            [pos[0, 1], dir_vec_y[1]],
            [pos[0, 2], dir_vec_y[2]],
            color=color[1],
            label=label[1],
        )
        ax.plot(
            [pos[0, 0], dir_vec_z[0]],
            [pos[0, 1], dir_vec_z[1]],
            [pos[0, 2], dir_vec_z[2]],
            color=color[2],
            label=label[2],
        )
    else:
        ax.plot([pos[0, 0], dir_vec_x[0]], [pos[0, 1], dir_vec_x[1]], [pos[0, 2], dir_vec_x[2]], color=color[0])
        ax.plot([pos[0, 0], dir_vec_y[0]], [pos[0, 1], dir_vec_y[1]], [pos[0, 2], dir_vec_y[2]], color=color[1])
        ax.plot([pos[0, 0], dir_vec_z[0]], [pos[0, 1], dir_vec_z[1]], [pos[0, 2], dir_vec_z[2]], color=color[2])
    ax.text(pos[0, 0], pos[0, 1], pos[0, 2], text)

    label_every_nth_pose = 2
    # if num_poses / plot_every_nth_pose > 500:
    #   plot_every_nth_pose = int(num_poses / 500)
    #   print(plot_every_nth_pose)
    for k in range(1, num_poses):
        if k % plot_every_nth_pose != 0:
            continue
        dir_vec_x = pos[k] + x_mod * x_ori[k]
        dir_vec_y = pos[k] + y_mod * y_ori[k]
        dir_vec_z = pos[k] + z_mod * z_ori[k]
        ax.plot([pos[k, 0], dir_vec_x[0]], [pos[k, 1], dir_vec_x[1]], [pos[k, 2], dir_vec_x[2]], color=color[0])
        ax.plot([pos[k, 0], dir_vec_y[0]], [pos[k, 1], dir_vec_y[1]], [pos[k, 2], dir_vec_y[2]], color=color[1])
        ax.plot([pos[k, 0], dir_vec_z[0]], [pos[k, 1], dir_vec_z[1]], [pos[k, 2], dir_vec_z[2]], color=color[2])
        if k % label_every_nth_pose == 0:
            ax.text(pos[k, 0], pos[k, 1], pos[k, 2], str(k))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    min = np.min([np.min(pos[:, 1]), np.min(pos[:, 0]), np.min(pos[:, 2])])
    max = np.max([np.max(pos[:, 1]), np.max(pos[:, 0]), np.max(pos[:, 2])])
    ax.set_xlim([min, max])
    ax.set_ylim([min, max])
    ax.set_zlim([min, max])
    # plt.gca().invert_yaxis()

    if title is not None:
        if new_plot:
            plt.legend()
            plt.title(title)
            plt.show()
        else:
            plt.legend()
            plt.title(title)
    else:
        plt.legend()
    return ax     



def get_ray_direction(H: int, W: int, focal: float, use_pixel_centers: bool = True) -> Float[Tensor, "H W 3"]:
    """
    Get ray direction for all pixel in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0
    fx, fy = focal, focal
    cx, cy = W / 2, H / 2
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy"
    )
    # blender coordinate
    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1
    )
    return directions

def get_rays(
    directions: Float[Tensor, "B H W 3"],
    c2w: Float[Tensor, "B 4 4"],
    keepdim: bool = False,
    normalize: bool = True,
    noise_scale: float = 0.0
) -> Tuple[Float[Tensor, "B H W 3"], Float[Tensor, "B H W 3"]]:
    assert directions.shape[-1] == 3
    if directions.ndim == 3:
        directions = directions.unsqueeze(0)
        c2w = c2w.unsqueeze(0)
    rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(-1) # (B H W 3)
    rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    
    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale
    if normalize:
        rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    return rays_o, rays_d

def evalute_img(
    pred_imgs: Float[Tensor, "b i c h w"],
    gt_imgs: Float[Tensor, "b i c h w"]
):
    assert pred_imgs.shape == gt_imgs.shape
    _, b, c, h, w = pred_imgs.shape
    pred_imgs = pred_imgs.squeeze(0)
    gt_imgs = gt_imgs.squeeze(0)
    lpips_model = lpips.LPIPS(net="vgg").to(pred_imgs.device)
    psnr_data = torch.zeros(b)
    ssim_data = torch.zeros(b)
    lpips_data = torch.zeros(b)
    
    # lpips_data = lpips_model(gt_imgs, pred_imgs).item()
    
    # pred_imgs = (pred_imgs.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.).astype(np.uint8)
    # pred_imgs = cv.cvtColor(pred_imgs, cv.COLOR_RGB2BGR)
    # gt_imgs = (gt_imgs.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.).astype(np.uint8)
    # gt_imgs = cv.cvtColor(gt_imgs, cv.COLOR_RGB2BGR)
    for i in range(b):
        pred_img = pred_imgs[i]
        gt_img = gt_imgs[i]
        lpips_data[i] = lpips_model(pred_img, gt_img)
        
        # convert tensor to numpy 0 - 1 to 0 - 255
        pred_img = cv.cvtColor(
            (pred_img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8),
            cv.COLOR_RGB2BGR)
        gt_img = cv.cvtColor(
            (gt_img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8),
            cv.COLOR_RGB2BGR
        )
        psnr = peak_signal_noise_ratio(gt_img, pred_img)
        ssim = structural_similarity(gt_img, pred_img, multichannel=True, channel_axis=-1)
        psnr_data[i] = psnr
        ssim_data[i] = ssim
    
    return psnr_data.mean(), ssim_data.mean(), lpips_data.mean()
