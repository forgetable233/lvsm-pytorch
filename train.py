import os
import accelerate
import time
import datetime

import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse
import cv2 as cv
import numpy as np
import wandb
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from lvsm_pytorch import LVSM
from utils.data_utils import ScanNetDataset, ValidationWrapper, Re10kDatasetTest
from dataset.data_module import DataModule
from dataset.dataset_re10k import DatasetRe10kCfg, Re10kDataset, Re10kDatasetSmall
from config import load_typed_root_config, RootCfg

@hydra.main(version_base=None, config_path="./configs", config_name="base")
def main(cfg: DictConfig):
    # prepare log infor
    output_dir = cfg.logger.output_dir
    test_name = cfg.logger.name
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    test_name = f"{test_name}_{cur_time}"
    output_dir = os.path.join(output_dir, test_name)
    OmegaConf.update(cfg, "logger.output_dir", output_dir)
    if cfg.logger.log:
        os.makedirs(output_dir, exist_ok=True)
    # prepare model     
    model_params = cfg.model
    model = LVSM(
        model_params,
        output_dir=output_dir
    )

    DATASET = {
        "scannet": ScanNetDataset,
        "re10k": Re10kDatasetTest
    }
    # prepare dataset
    data_params = cfg.dataset
    train_scannet = DATASET[data_params.name](**data_params)
    val_dataset = ValidationWrapper(train_scannet, 1)
    train_loader = DataLoader(
        dataset=train_scannet,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    # prepare train params
    train_params = cfg.train
    if train_params.resume:
        assert os.path.exists(train_params.ckpt)
        model.load_state_dict(train_params.ckpt)
    
    if cfg.logger.log and cfg.logger.use_wandb:
        wandb_logger = WandbLogger(project=cfg.logger.wandb.project, name=cfg.logger.wandb.name)
        trainer = pl.Trainer(logger=wandb_logger, **train_params.trainer)
    else:
        trainer = pl.Trainer(**train_params.trainer)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()