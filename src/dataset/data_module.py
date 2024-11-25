import os
from typing import Iterator, Optional

import torch
from torch import Generator
from torch.utils.data import Dataset, IterableDataset, DataLoader
import pytorch_lightning as pl

import cv2 as cv
import numpy as np
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf


from .dataset_re10k import DatasetRe10kCfg, Re10kDataset
from .data_test import ScanNetDataset, Re10kDatasetTest
from config import RootCfg, DataLoaderCfg, DatasetRe10kCfg, DataLoaderStageCfg

DATASET = {
    "re10k": Re10kDataset,
    "re10k_small": Re10kDatasetTest,
    "scannet": ScanNetDataset
}

def get_dataset(cfg, stage):
    return DATASET[cfg.name](cfg, stage)

class DataModule(pl.LightningDataModule):
    data_loader_cfg: DataLoaderCfg
    global_rank: int
    
    def __init__(self,
                 cfg: RootCfg,
                 global_rank) -> None:
        super().__init__()
        self.data_loader_cfg = cfg.data_loader
        self.dataset_cfg = cfg.dataset
        self.global_rank = global_rank
        print("dataset loaded")
        
    def train_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "train")
        return DataLoader(
            dataset=dataset,
            batch_size=self.data_loader_cfg.train.batch_size,
            shuffle=not isinstance(dataset, IterableDataset),
            num_workers=self.data_loader_cfg.train.num_workers,
            generator=self.get_generator(self.data_loader_cfg.train),
            persistent_workers=self.get_persistent(self.data_loader_cfg.train)
        )
    
    def val_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "val")
        return DataLoader(
            dataset=ValidationWrapper(dataset, 1),
            batch_size=self.data_loader_cfg.val.batch_size,
            shuffle= not isinstance(dataset, IterableDataset),
            num_workers=self.data_loader_cfg.val.num_workers,
            generator=self.get_generator(self.data_loader_cfg.val),
            persistent_workers=self.get_persistent(self.data_loader_cfg.val)
        )    
    
    def test_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "test")
        return DataLoader(
            dataset=dataset,
            batch_size=self.data_loader_cfg.test.batch_size,
            shuffle= not isinstance(dataset, IterableDataset),
            num_workers=self.data_loader_cfg.test.num_workers,
            generator=self.get_generator(self.data_loader_cfg.test),
            persistent_workers=self.get_persistent(self.data_loader_cfg.test)
        )
    
    def get_generator(self, cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(cfg.seed + self.global_rank)
        return generator
    
    def get_persistent(self, cfg: DataLoaderStageCfg) -> bool | None:
        return True if cfg.num_workers != 1 else False
    

class ValidationWrapper(Dataset):
    dataset: Dataset
    dataset_iterator: Optional[Iterator]
    length: int

    def __init__(self, dataset: Dataset, length: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.length = length
        self.dataset_iterator = None

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        if isinstance(self.dataset, IterableDataset):
            if self.dataset_iterator is None:
                self.dataset_iterator = iter(self.dataset)
            return next(self.dataset_iterator)
        
        random_index = torch.randint(0, len(self.dataset), tuple())
        return self.dataset[random_index.item()]



