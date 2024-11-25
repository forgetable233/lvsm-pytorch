import os

import torch

def apply_augmentation_shim(
    example,
    generate: torch.Generator | None = None
):
    if torch.rand(tuple(), generator=generate) < 0.5:
        return example
    
