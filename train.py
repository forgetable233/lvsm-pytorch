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
from tqdm import tqdm

from lvsm_pytorch import LVSM
from utils.data_utils import ScanNetDataset

def main(args):
    output_dir = args.output_dir
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = os.path.join(output_dir, cur_time)
    
    
    
    
    dim = 256
    patch_size = 32
    depth = 6
    
    model = LVSM(
        dim=dim,
        max_image_size_width=640,
        max_image_size_height=480,
        patch_size=patch_size,
        depth=depth,
        max_input_images=4,
        output_dir=output_dir
    )
    
    wandb.init(
        project="lvsm",
        name=f"dim_{dim}_path_size_{patch_size}_depth_{depth}"
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
        batch_size=8,
        shuffle=False,
    )
    
    val_loader = DataLoader(
        dataset=train_scannet,
        batch_size=1,
        shuffle=False
    )
    
    epo = 1000
    
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=epo, check_val_every_n_epoch=20)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(args)
