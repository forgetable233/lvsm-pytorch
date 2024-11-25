import os
from typing import Iterator, Optional, Literal

import torch
from torch.utils.data import IterableDataset, Dataset
from torch import Tensor
from jaxtyping import Float

from dataclasses import dataclass

# from .view_sampler import ViewSamplerCfg


@dataclass
class DatasetCfgCommon:
    image_shape: list[int]
    background_color: list[float]
    cameras_are_circular: bool
    overfit_to_scene: str | None
    # view_sampler: ViewSamplerCfg


class DatasetBase(IterableDataset):
    cfg: DatasetCfgCommon
    stage: Literal["train", "val", "test"]
    
    def __init__(self,
                 cfg: DatasetCfgCommon,
                 stage):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
    
