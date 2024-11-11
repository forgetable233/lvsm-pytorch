import os
import accelerate
import time
import datetime

import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import pytorch_lightning as pl
import argparse
import cv2 as cv
import numpy as np
import wandb
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from lvsm_pytorch import LVSM
from utils.data_utils import ScanNetDataset

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
        
    
    model_params = cfg.model
   
   
    model = LVSM(
        model_params,
        output_dir=output_dir
    )
    if cfg.logger.use_wandb:
        wandb.init(
            project=cfg.logger.wandb.project,
            name=cfg.logger.wandb.name
        )
    
    train_scannet = ScanNetDataset(
        root="/run/determined/workdir/data/scannet/scans/scene0000_00/extract", 
        rgb_folder="color_resize",
        K_name="intrinsic_color_resize.txt"
        )
    
    # train_length = int(len(train_scannet) * 0.8)
    # val_length = len(train_scannet) - train_length
    # seed = torch.Generator().manual_seed(42)
    # train_set, valid_set = data.random_split(train_scannet, [train_length, val_length], generator=seed)
    
    train_loader = DataLoader(
        dataset=train_scannet,
        batch_size=1,
        shuffle=False,
    )
    
    val_loader = DataLoader(
        dataset=train_scannet,
        batch_size=1,
        shuffle=False
    )
    
    epo = 2000
    
    trainer = pl.Trainer(limit_train_batches=600, max_epochs=epo, check_val_every_n_epoch=100)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--bs", type=int, default=32)
    # parser.add_argument("--output_dir", type=str, default="outputs")
    # args = parser.parse_args()
    main()
