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

from src.lvsm_pytorch import LVSM
from src.dataset.data_test import ScanNetDataset, Re10kDatasetTest
from src.dataset.data_module import DataModule, ValidationWrapper
from src.dataset.dataset_re10k import DatasetRe10kCfg, Re10kDataset, Re10kDatasetSmall
from src.config import load_typed_root_config, RootCfg

DATASET = {
        "scannet": ScanNetDataset,
        "re10k_small": Re10kDatasetTest,
        "re10k": Re10kDataset
    }

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

    # unpack configs
    OmegaConf.resolve(cfg)
    cfg: RootCfg = load_typed_root_config(cfg)
    
    # prepare trainer
    train_params = cfg.train
    if cfg.logger.log and cfg.logger.use_wandb:
        wandb_logger = WandbLogger(project=cfg.logger.wandb.project, name=cfg.logger.wandb.name)
        trainer = pl.Trainer(logger=wandb_logger, 
                             default_root_dir=output_dir, 
                             **vars(train_params.trainer))
    else:
        trainer = pl.Trainer(**vars(train_params.trainer))
    
    # load dataset
    data_module = DataModule(cfg, trainer.global_rank)
    
    # prepare model     
    model_params = cfg.model
    model = LVSM(
        model_params,
        output_dir=output_dir
    )
        
    # start to train the model
    if train_params.resume:
        assert os.path.exists(train_params.ckpt)
        trainer.fit(model, datamodule=data_module, ckpt_path=train_params.ckpt)
    else:
        trainer.fit(model, datamodule=data_module)
        
    exit()
    # train_scannet = Re10kDataset(cfg.dataset, "train")
    # print(len(train_scannet))
    # train_scannet = DATASET[data_params.name](**data_params)
    val_dataset = ValidationWrapper(train_scannet, 1)
    train_loader = DataLoader(
        dataset=train_scannet,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        persistent_workers=False
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
        trainer = pl.Trainer(logger=wandb_logger, default_root_dir=output_dir, **train_params.trainer)
    else:
        trainer = pl.Trainer(**train_params.trainer)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()