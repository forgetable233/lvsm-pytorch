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
        if stage in ["train", "val"]:
            assert os.path.basename(self.cfg.root) == "train"
        else:
            assert os.path.basename(self.cfg.root) == "test"
        
        for chunk in sorted(os.listdir(self.cfg.root)):
            chunk_path = os.path.join(self.cfg.root, chunk)
            self.chunks.append(chunk_path)
        
    
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
        