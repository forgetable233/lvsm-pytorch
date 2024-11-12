import os

import torch
import pytorch_lightning as pl
from lightning.pytorch import LightningDataModule

import cv2 as cv
import numpy as np
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf


from .dataset_re10k import DatasetRe10kCfg

@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None

@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg

class DataModule(LightningDataModule):
    dataset_cfg: DatasetRe10kCfg
    data_loader_cfg: DataLoaderCfg
    global_rank: int
    
    def __init__(self) -> None:
        super().__init__()
        
    def train_dataloader(self):
        return super().train_dataloader()
    
    def val_dataloader(self):
        return super().val_dataloader()
    
    def test_dataloader(self):
        return super().test_dataloader()
    
    



