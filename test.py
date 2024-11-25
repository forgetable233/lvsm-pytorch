import os

import torch
from torch.utils.data import IterableDataset


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from src.utils.geometry_utils import plot_cooridinate_c2w

class testIterableDataset(IterableDataset):
    def __init__(self, length) -> None:
        super().__init__()
        self.length = length
        
    def __len__(self):
        return self.length
    
    def __iter__(self):
        for i in range(self.length):
            yield i

a = testIterableDataset(10)
a_iter = iter(a)
print(next(a_iter))