from __future__ import annotations
from lvsm_pytorch.tensor_typing import *

import os
from functools import wraps

import torchvision

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import numpy as np
import cv2 as cv

import wandb

from x_transformers import Encoder

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, pack, unpack

"""d
ein notation:
b - batch
n - sequence
h - height
w - width
c - channels (either 6 for plucker rays or 3 for rgb)
i - input images
"""

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def lens_to_mask(lens: Int[Tensor, 'b'], max_length: int):
    seq = torch.arange(max_length, device = lens.device)
    return einx.less('b, n -> b n', lens, seq)

def divisible_by(num, den):
    return (num % den) == 0

# class

class LVSM(pl.LightningModule):
    def __init__(
        self,
        dim,
        max_image_size_width,
        max_image_size_height,
        patch_size,
        depth = 12,
        heads = 8,
        max_input_images = 32,
        dim_head = 64,
        channels = 3,
        rand_input_image_embed = True,
        decoder_kwargs: dict = dict(
            use_rmsnorm = True,
            add_value_residual = True,
            ff_glu = True,
        ),
        perceptual_loss_weight = 0.5,    # they use 0.5 for scene-level, 1.0 for object-level
        output_dir: str = "./outputs"
    ):
        super().__init__()
        assert divisible_by(max_image_size_width, patch_size)
        assert divisible_by(max_image_size_height, patch_size)
        # prepare output path
        self.ckpt_path = os.path.join(output_dir, "ckpt")
        self.img_path = os.path.join(output_dir, "img")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.img_path, exist_ok=True)
        
        
        # positional embeddings

        self.width_embed = nn.Parameter(torch.zeros(max_image_size_width // patch_size, dim))
        self.height_embed = nn.Parameter(torch.zeros(max_image_size_height // patch_size, dim))
        self.input_image_embed = nn.Parameter(torch.zeros(max_input_images, dim))

        nn.init.normal_(self.width_embed, std = 0.02)
        nn.init.normal_(self.height_embed, std = 0.02)
        nn.init.normal_(self.input_image_embed, std = 0.02)

        self.rand_input_image_embed = rand_input_image_embed

        # raw data to patch tokens for attention

        patch_size_sq = patch_size ** 2

        self.input_to_patch_tokens = nn.Sequential(
            Rearrange('b i c (h p1) (w p2) -> b i h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear((6 + channels) * patch_size_sq, dim)
        )

        self.target_rays_to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(6 * patch_size_sq, dim)
        )

        self.decoder = Encoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            **decoder_kwargs
        )

        self.target_unpatchify_to_image = nn.Sequential(
            nn.Linear(dim, channels * patch_size_sq),
            nn.Sigmoid(),
            Rearrange('b h w (c p1 p2) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size, c = channels)
        )

        self.has_perceptual_loss = perceptual_loss_weight > 0. and channels == 3
        self.perceptual_loss_weight = perceptual_loss_weight

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # for tensor typing

        self._c = channels

    @property
    def device(self):
        return self.zero.device

    @property
    def vgg(self):

        if not self.has_perceptual_loss:
            return None

        if hasattr(self, '_vgg'):
            return self._vgg[0]

        vgg = torchvision.models.vgg16(pretrained = True)
        vgg.classifier = nn.Sequential(*vgg.classifier[:-2])
        vgg.requires_grad_(False)

        self._vgg = [vgg]
        return vgg.to(self.device)
    
    def forward(
        self,
        input_images: Float[Tensor, 'b i {self._c} h w'],
        input_rays: Float[Tensor, 'b i 6 h w'],
        target_rays: Float[Tensor, 'b 6 h w'],
        target_images: Float[Tensor, 'b {self._c} h w'] | None = None,
        num_input_images: Int[Tensor, 'b'] | None = None,
        return_loss_breakdown = False
    ):

        input_tokens = self.input_to_patch_tokens(torch.cat((input_images, input_rays), dim = -3))

        target_tokens = self.target_rays_to_patch_tokens(target_rays)

        # add positional embeddings

        _, num_images, height, width, _ = input_tokens.shape

        height_embed = self.height_embed[:height]
        width_embed = self.width_embed[:width]

        input_tokens = einx.add('b i h w d, h d, w d -> b i h w d', input_tokens, height_embed, width_embed)

        target_tokens = einx.add('b h w d, h d, w d -> b h w d', target_tokens, height_embed, width_embed)

        # add input image embeddings, make it random to prevent overfitting

        if self.rand_input_image_embed:
            batch, max_num_input_images = input_tokens.shape[0], self.input_image_embed.shape[0]

            randperm = torch.randn((batch, max_num_input_images), device = self.device).argsort(dim = -1)
            randperm = randperm[:, :num_images]

            rand_input_image_embed = self.input_image_embed[randperm]

            input_tokens = einx.add('b i h w d, b i d -> b i h w d', input_tokens, rand_input_image_embed)
        else:
            input_image_embed = self.input_image_embed[:num_images]
            input_tokens = einx.add('b i h w d, i d -> b i h w d', input_tokens, input_image_embed)

        # pack dimensions to ready for attending

        input_tokens, _ = pack([input_tokens], 'b * d')
        target_tokens, packed_height_width = pack([target_tokens], 'b * d')

        tokens, packed_shape = pack([target_tokens, input_tokens], 'b * d')

        # take care of variable number of input images

        mask = None

        if exists(num_input_images):
            mask = lens_to_mask(num_input_images, num_images + 1) # plus one for target patched rays
            mask = repeat(mask, 'b i -> b (i hw)', hw = height * width)

        # attention

        tokens = self.decoder(tokens, mask = mask)

        # unpack

        target_tokens, input_tokens = unpack(tokens, packed_shape, 'b * d')

        # project target tokens out

        target_tokens, = unpack(target_tokens, packed_height_width, 'b * d')

        # project back to image

        pred_target_images = self.target_unpatchify_to_image(target_tokens)

        if not exists(target_images):
            return pred_target_images

        loss =  F.mse_loss(pred_target_images, target_images)

        self.log("train/mes_loss", loss.item())
        perceptual_loss = self.zero

        if self.has_perceptual_loss:
            self.vgg.eval()

            target_image_vgg_feats = self.vgg(target_images)
            pred_target_image_vgg_feats = self.vgg(pred_target_images)

            perceptual_loss = F.mse_loss(target_image_vgg_feats, pred_target_image_vgg_feats)
            self.log("train/perceptual_loss", perceptual_loss.item())

        total_loss = (
            loss +
            perceptual_loss * self.perceptual_loss_weight
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, perceptual_loss)
    
    def training_step(self, batch, batch_idx):
        rgb = batch["rgb"]
        rays = batch["rays"]
        target_rgb = batch["target_rgb"]
        target_rays = batch["target_rays"]
        loss = self.forward(
            input_images=rgb,
            input_rays=rays,
            target_rays=target_rays,
            target_images=target_rgb
        )
        self.log("train/total_loss", loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        rgb = batch["rgb"]
        rays = batch["rays"]
        target_rgb = batch["target_rgb"]
        target_rays = batch["target_rays"]
        val_rgb = self.forward(
            input_images=rgb,
            input_rays=rays,
            target_rays=target_rays,
        )
        _, _, c, h, w = rgb.shape
        output_rgb = np.zeros((h, w * 2, c), dtype=np.uint8)
        target_rgb = (target_rgb.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
        val_rgb = (val_rgb.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
        
        output_rgb[:, :w, :] = target_rgb
        output_rgb[:, w:, :] = val_rgb
        output_rgb = cv.cvtColor(output_rgb, cv.COLOR_RGB2BGR)
        torch.save(self.state_dict(), os.path.join(self.ckpt_path, f"{self.global_step}.pt"))
        cv.imwrite(os.path.join(self.img_path, f"{self.global_step}.jpg"), output_rgb)
        # self.log(f"img/{self.global_step}", output_rgb)
        return val_rgb

    @rank_zero_only
    def log(self, name, item):
        wandb.log({name: item})
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
