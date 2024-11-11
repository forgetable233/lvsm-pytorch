import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np

from lvsm_pytorch.tensor_typing import *

class ScanNetDataset(Dataset):
    
    def __init__(self,
                 root,
                 rgb_folder: str = "color",
                 pose_folder: str = "pose",
                 K_name: str = "intrinsic_color.txt",
                 img_num: int = 4,
                 width: int = 256,
                 height: int = 256,
                 stage: str = "train",
                 **kwargs) -> None:
        super().__init__()
        self.root = root
        self.rgb_folder = os.path.join(root, rgb_folder)
        self.pose_folder = os.path.join(root, pose_folder)
        self.K: Float[Tensor, "3 3"] = torch.from_numpy(np.loadtxt(os.path.join(root, "intrinsic", K_name)))
        
        self.width, self.height = width, height
        
        self.len = len(os.listdir(self.rgb_folder))
        self.img_h, self.img_w, _ = cv.imread(os.path.join(self.rgb_folder, "0.jpg")).shape
        self.resize_K()
        
        self.fovy = 2 * torch.atan(self.height / (2 * self.K[1, 1]))
        self.fovx = 2 * torch.atan(self.width / (2 * self.K[0, 0]))
        
        self.focal_length = torch.tensor([0.5 * self.height / torch.tan(0.5 * self.fovy)])
        self.directions_unit_focals: Float[Tensor, "H W 3"] = get_ray_direction(self.height, self.width, focal=1.0)
        
        self.directions: Float[Tensor, "H W 3"] = self.directions_unit_focals.clone()
        self.directions[:, :, :2] = self.directions[:, :, :2] / self.focal_length[:, None, None]
        
        self.img_num = img_num
        self.rgb_path = []
        self.pose_path = []
        for i in range(self.len):
            if rgb_folder != "color_resize":
                rgb_path = os.path.join(self.rgb_folder, f"{i}.jpg")
                pose_path = os.path.join(self.pose_folder, f"{i}.txt")
            else:
                rgb_path = os.path.join(self.rgb_folder, f"{int(i * 5)}.jpg")
                pose_path = os.path.join(self.pose_folder, f"{int(i * 5)}.txt")
            assert os.path.exists(rgb_path)
            assert os.path.exists(pose_path)
            self.rgb_path.append(rgb_path)
            self.pose_path.append(pose_path)
        
    def resize_K(self):
        alpha_y = self.height / self.img_h
        alpha_x = self.width / self.img_w
        self.K[0, 0] *= alpha_x
        self.K[0, 2] *= alpha_x
        self.K[1, 1] *= alpha_y
        self.K[1, 2] *= alpha_y
    
    def __len__(self):
        return self.len
        
    def __getitem__(self, index) -> Any:
        if index + self.img_num >= self.len:
            index = self.len - 2 - self.img_num
        
        rgb: Float[Tensor, "i 3 H W"] = torch.from_numpy(
            np.stack([cv.cvtColor(cv.resize(cv.imread(rgb_img), (self.width, self.height)), cv.COLOR_BGR2RGB) for rgb_img in self.rgb_path[index:index + self.img_num]])).permute(0, 3, 1, 2)
        rgb = rgb / 255.
        pose: Float[Tensor, "i 4 4"] = torch.from_numpy(np.stack([np.loadtxt(pose_file) for pose_file in self.pose_path[index:index + self.img_num]]))
        rays = torch.zeros((self.img_num, self.height, self.width, 6))
        for i in range(pose.shape[0]):
            rays_o, rays_d = get_rays(
                self.directions, pose[i], keepdim=True, normalize=True
            )
            rays[i, :, :, :3] = rays_o
            rays[i, :, :, 3:] = rays_d
        rays = rays.permute(0, 3, 1, 2)
        
        target_rgb = torch.from_numpy(cv.cvtColor(cv.resize(cv.imread(self.rgb_path[index + self.img_num]), (self.width, self.height)), cv.COLOR_BGR2RGB)).permute(2, 0, 1)
        target_rgb = target_rgb / 255.
        target_pose = torch.from_numpy(np.loadtxt(self.pose_path[index + self.img_num]))
        tar_rays = torch.zeros((self.height, self.width, 6))
        tar_rays[:, :, :3], tar_rays[:, :, 3:] = get_rays(
            self.directions, target_pose, keepdim=True, normalize=True
        )
        tar_rays = tar_rays.permute(2, 0, 1)
        # target_rgb: Float[Tensor, "3 H W"] = torch.from_numpy()
        return {
            "rgb": rgb,
            "pose": pose,
            "K": self.K,
            "rays": rays,
            "target_rgb": target_rgb,
            "target_rays": tar_rays
        }
    
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
    # opencv coordinate
    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1
    )
    return directions

def get_rays(
    directions: Float[Tensor, "H W 3"],
    c2w: Float[Tensor, "4 4"],
    keepdim: bool = False,
    normalize: bool = True,
    noise_scale: float = 0.0
) -> Tuple[Float[Tensor, "H W 3"], Float[Tensor, "H W 3"]]:
    assert directions.shape[-1] == 3
    rays_d = (directions[:, :, None, :]* c2w[None, None, :3, :3]).sum(-1) # (H W 3)
    rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
    
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
    
