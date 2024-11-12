import os
import torch
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

from .dataset import DatasetCfgCommon

@dataclass
class DatasetRe10kCfg:
    name: Literal["re10k"]
    width: int
    height: int
    batch_size: int
    img_num: int
    data_params: str

class Re10kDataset:
    cfg: DatasetRe10kCfg
    stage: str
    
    def __init__(
        self,
        cfg: DatasetRe10kCfg,
        stage: str,
        **kwargs) -> None:
        
        pass