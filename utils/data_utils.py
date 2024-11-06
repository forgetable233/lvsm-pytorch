import os
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np

class ScanNetDataset(Dataset):
    
    def __init__(self,
                 root,
                 device) -> None:
        super().__init__()
        self.device = device
        self.root = root
        self.rgb_folder = os.path.join(root, "color")
        self.pose_folder = os.path.join(root, "pose")
        self.K = torch.from_numpy(np.loadtxt(os.path.join(root, "intrinsic", "intrinsic_color.txt"))).to(device)
        self.len = len(os.listdir(self.rgb_folder))
        h, w, c = cv.imread(os.path.join(self.rgb_folder, "0.jpg")).shape
        self.rgb_path = []
        self.pose_path = []
        for i in range(self.len):
            rgb_path = os.path.join(self.rgb_folder, f"{i}.jpg")
            pose_path = os.path.join(self.pose_folder, f"{i}.txt")
            assert os.path.exists(rgb_path)
            assert os.path.exists(pose_path)
            self.rgb_path.append(rgb_path)
            self.pose_path.append(pose_path)
    
    def __len__(self):
        return self.len
        
    def __getitem__(self, index) -> Any:
        rgb = torch.from_numpy(cv.cvtColor(cv.imread(self.rgb_path[index]), cv.COLOR_BGR2RGB)).to(self.device).permute(2, 0, 1)
        rgb = rgb / 127.5 - 1
        pose = torch.from_numpy(np.loadtxt(self.pose_path[index])).to(self.device)
        
        return {
            "rgb": rgb,
            "pose": pose,
            "K": self.K
        }
