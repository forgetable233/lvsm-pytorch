import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import IterableDataset
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image

from .dataset import DatasetCfgCommon
from .view_sampler.view_sampler_scene import ViewSamplerSceneCfg
from utils.geometry_utils import get_fov

@dataclass
class DatasetRe10kCfg:
    name: Literal["re10k"]
    width: int
    height: int
    batch_size: int
    img_num: int
    data_params: str
    root: str
    small_test: bool
    max_fov: float
    make_baseline_1: bool
    
    view_sampler: ViewSamplerSceneCfg

class Re10kDataset(IterableDataset):
    cfg: DatasetRe10kCfg
    stage: Literal["train", "val", "test"]
    
    chunks: list[Path]
    
    def __init__(
        self,
        cfg: DatasetRe10kCfg,
        stage: str,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        
        assert os.path.exists(self.cfg.root)
        self.chunks = []
        if self.stage in ["train", "val"]:
            assert os.path.basename(self.cfg.root) == "train"
        else:
            assert os.path.basename(self.cfg.root) == "test"
        chunk_files = sorted([file for file in os.listdir(self.cfg.root) if file.endswith(".torch")])
        # collect chunks
        for chunk in chunk_files:
            chunk_path = os.path.join(self.cfg.root, chunk)
            self.chunks.append(chunk_path)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]
    
    def __iter__(self):
        if self.stage in ["train", "val"]:
            self.chunks = self.shuffle(self.chunks)
        
        for chunk_path in self.chunks:
            chunk = torch.load(chunk_path)
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
                
                # TODO convert the example to ray and resize the image
                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        "near": self.get_bound("near", len(context_indices)) / scale,
                        "far": self.get_bound("far", len(context_indices)) / scale,
                        "index": context_indices,
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "near": self.get_bound("near", len(target_indices)) / scale,
                        "far": self.get_bound("far", len(target_indices)) / scale,
                        "index": target_indices,
                    },
                    "scene": scene,
                }
            pass
    
    def __len__(self):
        return len(self.chunks)
    
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
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)
    
    def __iter__(self):
        
        for chunk_path in self.chunks:
                chunk = torch.load(chunk_path)
                ## TODO add shuffle for train and val current only use one scene for test
                for example in chunk:
                    extrinsics, intrinsics = self.convert_poses(example["cameras"])
        pass
        return super().__iter__()
    
    def __len__(self):
        return len(self.chunks)

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
    
    def __len__(self):
        return len(self.chunks)
        