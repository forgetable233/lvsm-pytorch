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
from utils.data_utils import ScanNetDataset
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
    OmegaConf.resolve(cfg)
    rootcfg: RootCfg = load_typed_root_config(cfg)
    testDataset = Re10kDatasetSmall(rootcfg.dataset, "train")
    print(len(testDataset))
    exit()
    # prepare model     
    model_params = cfg.model
    model = LVSM(
        model_params,
        output_dir=output_dir
    )
    
    # prepare dataset
    data_params = cfg.data
    train_scannet = ScanNetDataset(**data_params)
    train_loader = DataLoader(
        dataset=train_scannet,
        batch_size=32,
        shuffle=False,
        num_workers=16
    )
    
    val_loader = DataLoader(
        dataset=train_scannet,
        batch_size=32,
        shuffle=False,
        num_workers=16
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
