import os
import accelerate

import torch
from torch.utils.data import DataLoader
import argparse
import cv2 as cv
import numpy as np

from lvsm_pytorch import LVSM
from utils.data_utils import ScanNetDataset

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    scannet = ScanNetDataset(root="/run/determined/workdir/data/scannet/scans/scene0000_00/extract", device=device)
    
    model = LVSM(
        dim=512,
        max_image_size=1296,
        patch_size=16,
        depth=6,
        max_input_images=4
    ).to(device, dtype=torch.float32)
    
    loader = DataLoader(
        dataset=scannet,
        batch_size=32,
        shuffle=False
    )
    
    for batch in loader:
        rgb = batch["rgb"]
        rays = batch["rays"]
        target_rgb = batch["target_rgb"]
        target_rays = batch["target_rays"]
        loss = model(
            input_images = rgb,
            input_rays = rays,
            target_rays = target_rays,
            target_images = target_rgb
        )
        exit()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
