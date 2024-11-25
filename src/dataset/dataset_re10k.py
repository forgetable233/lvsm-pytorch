import os
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, List
import json

import torch
from torch import Tensor
from torch.utils.data import IterableDataset
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
import numpy as np

from .dataset import DatasetCfgCommon
from .view_sampler.view_sampler_scene import ViewSamplerSceneCfg, ViewSamplerScene
from utils.geometry_utils import get_fov, get_ray_direction, get_rays
from shims.argument_shim import apply_augmentation_shim
from shims.crop_shims import apply_crop_shim

@dataclass
class DatasetRe10kCfg:
    name: Literal["re10k"]
    shape: list[int, int]
    batch_size: int
    img_num: int
    data_params: str
    root: str
    small_test: bool
    max_fov: float
    make_baseline_1: bool
    
    baseline_epsilon: float
    augment: bool
    
    normalize: bool
    keepdim: bool
    noise_scale: float
    
    bf16: bool
    
    view_sampler: ViewSamplerSceneCfg

class Re10kDataset(IterableDataset):
    cfg: DatasetRe10kCfg
    stage: Literal["train", "val", "test"]
    
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0
    
    def __init__(
        self,
        cfg: DatasetRe10kCfg,
        stage: str,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        
        # collect chunks
        assert os.path.exists(self.cfg.root)
        self.chunks = []
        if self.stage in ["train", "val"]:
            root = os.path.join(self.cfg.root, "train")
        else:
            root = os.path.join(self.cfg.root, "test")
        assert os.path.exists(root)
        chunk_files = sorted([file for file in os.listdir(root) if file.endswith(".torch")])
        for chunk in chunk_files:
            chunk_path = os.path.join(root, chunk)
            self.chunks.append(chunk_path)
        
        # define view sampler
        self.view_sampler = ViewSamplerScene(self.cfg.view_sampler, stage=self.stage)
        
        # define directions
        self.directions_unit_focals: Float[Tensor, "H W 3"] = get_ray_direction(self.cfg.shape[0], self.cfg.shape[1], focal=1.0)
        if self.cfg.bf16:
            self.directions_unit_focals = self.directions_unit_focals.float()
        
        self.to_tensor = tf.ToTensor()
        

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]
    
    def __iter__(self):
        if self.stage in ["train", "val"]:
            self.chunks = self.shuffle(self.chunks)
        
        for chunk_path in self.chunks:
            chunk = torch.load(chunk_path, weights_only=True)
            if self.stage in ["train", "val"]:
                chunk = self.shuffle(chunk)
            
            for example in chunk:
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]
                
                # sample images from patch
                try:
                    context_indices, target_indices = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics
                    )
                except ValueError:
                    continue
                
                # skip the example if the field of view is too large
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue
                
                # load images
                try:
                    context_images = [
                        example["images"][index.item()] for index in context_indices
                    ]
                    context_images = self.convert_images(context_images)
                    target_images = [
                        example["images"][index.item()] for index in target_indices
                    ]
                    target_images = self.convert_images(target_images)
                except IndexError:
                    continue
                
                # skip when image do not have right shape
                context_image_invalid = context_images.shape[1:] != (3, 360, 640)
                target_image_invalid = target_images.shape[1:] != (3, 360, 640)
                if context_image_invalid or target_image_invalid:
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}."
                    )
                    continue
                
                # Resize the world to make the baseline 1
                context_extrinsics = extrinsics[context_indices]
                if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                    a, b = context_extrinsics[:, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_epsilon:
                        print(
                            f"Skipped {scene} because of insufficient baseline "
                            f"{scale:.6f}"
                        )
                        continue
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1
                
                # convert to float
                if not self.cfg.bf16:
                    example = {
                        "context": {
                            "extrinsics": extrinsics[context_indices],
                            "intrinsics": intrinsics[context_indices],
                            "images": context_images,
                            "near": self.get_bound("near", len(context_indices)) / scale,
                            "far": self.get_bound("far", len(context_indices)) / scale,
                            "index": context_indices,
                            "rays": self.get_rays(extrinsics[context_indices], 
                                                intrinsics[context_indices]),
                        },
                        "target": {
                            "extrinsics": extrinsics[target_indices],
                            "intrinsics": intrinsics[target_indices],
                            "images": target_images,
                            "near": self.get_bound("near", len(target_indices)) / scale,
                            "far": self.get_bound("far", len(target_indices)) / scale,
                            "index": target_indices,
                            "rays": self.get_rays(extrinsics[target_indices],
                                                intrinsics[target_indices]),
                        },
                        "scene": scene,
                    }
                else:
                    extrinsics = extrinsics.float()
                    intrinsics = intrinsics.float()
                    example = {
                        "context": {
                            "extrinsics": extrinsics[context_indices],
                            "intrinsics": intrinsics[context_indices],
                            "images": context_images,
                            "near": self.get_bound("near", len(context_indices)) / scale,
                            "far": self.get_bound("far", len(context_indices)) / scale,
                            "index": context_indices,
                            "rays": self.get_rays(extrinsics[context_indices], 
                                                intrinsics[context_indices]),
                        },
                        "target": {
                            "extrinsics": extrinsics[target_indices],
                            "intrinsics": intrinsics[target_indices],
                            "images": target_images,
                            "near": self.get_bound("near", len(target_indices)) / scale,
                            "far": self.get_bound("far", len(target_indices)) / scale,
                            "index": target_indices,
                            "rays": self.get_rays(extrinsics[target_indices],
                                                intrinsics[target_indices]),
                        },
                        "scene": scene,
                    }
                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)

                resized_exampe = apply_crop_shim(example, self.cfg.shape)
                yield resized_exampe
    
    def get_rays(self, 
                 extrinsics: Float[Tensor, "B 4 4"], 
                 intrinsics: Float[Tensor, "B 3 3"]
                 ) -> Float[Tensor, "B 6 H W"]:
        """
        get the pluker rays
        """
        b, _, _ = extrinsics.shape
        fov = 2 * torch.atan(self.cfg.shape[0] / (2 * intrinsics[:, 1, 1]))
        focal_length = 0.5 * self.cfg.shape[0] / torch.tan(0.5 * fov)
        directions = self.directions_unit_focals.unsqueeze(0).repeat(b, 1, 1, 1)
        directions[..., :2] /= focal_length[:, None, None, None]
        
        rays_o, rays_d = get_rays(directions, 
                                  extrinsics, 
                                  self.cfg.keepdim, 
                                  self.cfg.normalize, 
                                  self.cfg.noise_scale)
        moment = torch.linalg.cross(rays_o, rays_d)
        rays = torch.concat([moment, rays_d], dim=-1)
        return rays.permute(0, 3, 1, 2)
    
    def __len__(self):
        return len(self.chunks)
    
    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)
    
    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics                

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_image = self.to_tensor(image)
            torch_images.append(torch_image)
        return torch.stack(torch_images)
    
    # @cached_property
    # def index(self) -> dict[str, Path]:
    #     merged_index = {}
    #     data_stages = [self.data_stage]
    #     if self.cfg.overfit_to_scene is not None:
    #         data_stages = ("test", "train")
    #     for data_stage in data_stages:
    #         for root in self.cfg.root:
    #             # Load the root's index.
    #             with (os.path.join(root / data_stage / "index.json")).open("r") as f:
    #                 index = json.load(f)
    #             index = {k: Path(root / data_stage / v) for k, v in index.items()}

    #             # The constituent datasets should have unique keys.
    #             assert not (set(merged_index.keys()) & set(index.keys()))

    #             # Merge the root's index into the main index.
    #             merged_index = {**merged_index, **index}
    #     return merged_index
    

class Re10kDatasetSmall(Re10kDataset):
    cfg: DatasetRe10kCfg
    stage: Literal["train", "val", "test"]
    
    chunks: list[Path]
    def __init__(
        self,
        cfg: DatasetRe10kCfg,
        stage: str,
        ) -> None:
        super().__init__(cfg, stage)
        assert self.cfg.small_test
    
    def __iter__(self):
        """
        a small test to ensure model function
        """
        chunk = torch.load(self.chunks[0], weights_only=True)
        example = chunk[0]
        extrinsics, intrinsics = self.convert_poses(example["cameras"])
        scene = example["key"]
        
        return super().__iter__()
    
    def __len__(self) -> int:
        return len(self.chunks())
        