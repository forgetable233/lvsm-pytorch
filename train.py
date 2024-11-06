import os
import accelerate
import time
import datetime

import torch
from torch.utils.data import DataLoader
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
    
    ckpt_path = os.path.join(output_dir, "ckpt")
    img_path = os.path.join(output_dir, "img")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    scannet = ScanNetDataset(
        root="/run/determined/workdir/data/scannet/scans/scene0000_00/extract", 
        device=device,
        rgb_folder="color_resize",
        K_name="intrinsic_color_resize.txt")
    
    model = LVSM(
        dim=256,
        max_image_size_width=640,
        max_image_size_height=480,
        patch_size=32,
        depth=6,
        max_input_images=4
    ).to(device, dtype=torch.float32)
    
    loader = DataLoader(
        dataset=scannet,
        batch_size=8,
        shuffle=False,
    )
    
    opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    
    epo = 1000
    
    for i in tqdm(range(epo), desc="training"):
        for batch in loader:
            model.train()
            rgb = batch["rgb"].to(device)
            rays = batch["rays"].to(device)
            target_rgb = batch["target_rgb"].to(device)
            target_rays = batch["target_rays"].to(device)
            loss = model(
                input_images = rgb,
                input_rays = rays,
                target_rays = target_rays,
                target_images = target_rgb
            )
            loss.backward()
            opt.step()
            opt.zero_grad()
        if i % 5 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_path, f"{i}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(args)
