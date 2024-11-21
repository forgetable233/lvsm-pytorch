import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from src.utils.geometry_utils import plot_cooridinate_c2w

root = "./data/re10k/poses"

pose_files = sorted(os.listdir(root))
poses = np.zeros((len(pose_files), 4, 4))

for i, file in enumerate(pose_files):
    poses[i] = np.loadtxt(os.path.join(root, file))

img = plot_cooridinate_c2w(poses[:, :3, :])

plt.savefig("./temp/c2w.png")
