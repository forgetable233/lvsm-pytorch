import os
from PIL import Image
from io import BytesIO

import torch
import cv2 as cv
import numpy as np
from einops import rearrange

chunk = torch.load("/run/determined/workdir/data/re10k/train/000000.torch")
example = chunk[0]
poses = example["cameras"]
imgs = example["images"]
# assert poses.shape[0] == len(imgs)

len = poses.shape[0]

for i in range(len):
    img = np.array(Image.open(BytesIO(imgs[i].numpy().tobytes())))
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite(f"./data/re10k/imgs/{str(i).zfill(4)}.png", img)
    pose = poses[i, 6:]
    pose = rearrange(pose, "(h w) -> h w", h=3, w=4).numpy()
    c2w = np.eye(4)
    c2w[:3, :] = pose
    np.savetxt(f"./data/re10k/poses/{str(i).zfill(4)}.txt", c2w)
    k = np.eye(3)
    fx, fy, cx, cy = poses[i, :4].T
    print(fx, fy, cx, cy)
    k[0, 0] = fx
    k[0, 2] = cx
    k[1, 1] = fy
    k[1, 2] = cy
    np.savetxt(f"./data/re10k/intrinsic/{str(i).zfill(4)}.txt", k)
    


for key in example.keys():
    print(key)