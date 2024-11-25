import os

import torch
from torch import Tensor

from jaxtyping import Float


def recale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: list[int, int]
) -> tuple[
    Float[Tensor, "*#batch c h w"],
    Float[Tensor, "*#batc 3 3"]
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    
    ## resize and relocate the image
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2
    
    ## Center-crop the image
    images = images[..., :, row : row + h_out, col : col + w_out]
    
    ## adjust the intrinsics to account for cropping
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics
    

def apply_crop_shim_to_vies(views, shape: list[int, int]) :
    images, intrinsics = recale_and_crop(
        views["images"],
        views["intrinsics"],
        shape
    )
    return {
        **views,
        "images": images,
        "intrinsics": intrinsics
    }

def apply_crop_shim(example, shape: list[int, int]):
    return {
        **example,
        "context": apply_crop_shim(example["context"], shape),
        "target": apply_crop_shim(example["target", shape])
    }
