from __future__ import annotations
from lvsm_pytorch.tensor_typing import Float

from functools import wraps

import torchvision

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from x_transformers import Encoder

from einops.layers.torch import Rearrange
from einops import rearrange, repeat, pack, unpack

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, '_vgg')
        if has_vgg:
            vgg = self._vgg
            delattr(self, '_vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self._vgg = vgg

        return out
    return inner

# class

class LVSM(Module):
    def __init__(
        self,
        dim,
        *,
        patch_size,
        depth = 12,
        heads = 8,
        dim_head = 64,
        decoder_kwargs: dict = dict(),
        perceptual_loss_weight = 0.5    # they use 0.5 for scene-level, 1.0 for object-level
    ):
        super().__init__()

        self.input_to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(9 * patch_size ** 2, dim)
        )

        self.target_rays_to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(6 * patch_size ** 2, dim)
        )

        self.decoder = Encoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            **decoder_kwargs
        )

        self.target_unpatchify_to_image = nn.Sequential(
            nn.Linear(dim, 3 * patch_size ** 2),
            nn.Sigmoid(),
            Rearrange('b h w (c p1 p2) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size, c = 3)
        )

        self.has_perceptual_loss = perceptual_loss_weight > 0.
        self.perceptual_loss_weight = perceptual_loss_weight

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    @property
    def vgg(self):
        if hasattr(self, '_vgg'):
            return self._vgg

        vgg = torchvision.models.vgg16(pretrained = True)
        vgg.classifier = nn.Sequential(*vgg.classifier[:-2])
        self._vgg = vgg.to(self.device)
        return self._vgg

    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def forward(
        self,
        input_images: Float['b 3 h w'],
        input_rays: Float['b 6 h w'],
        target_rays: Float['b 6 h w'],
        target_images: Float['b 3 h w'] | None = None,
        return_loss_breakdown = False
    ):

        input_tokens = self.input_to_patch_tokens(torch.cat((input_images, input_rays), dim = 1))

        target_tokens = self.target_rays_to_patch_tokens(target_rays)

        input_tokens, _ = pack([input_tokens], 'b * d')
        target_tokens, packed_height_width = pack([target_tokens], 'b * d')

        tokens, packed_shape = pack([input_tokens, target_tokens], 'b * d')

        # attention

        tokens = self.decoder(tokens)

        # unpack

        input_tokens, target_tokens = unpack(tokens, packed_shape, 'b * d')

        # project target tokens out

        target_tokens, = unpack(target_tokens, packed_height_width, 'b * d')

        # project back to image

        pred_target_images = self.target_unpatchify_to_image(target_tokens)

        if not exists(target_images):
            return pred_target_images

        loss =  F.mse_loss(pred_target_images, target_images)

        perceptual_loss = self.zero

        if self.has_perceptual_loss:
            target_image_vgg_feats = self.vgg(target_images)
            pred_target_image_vgg_feats = self.vgg(pred_target_images)

            perceptual_loss = F.mse_loss(target_image_vgg_feats, pred_target_image_vgg_feats)

        total_loss = (
            loss +
            perceptual_loss * self.perceptual_loss_weight
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, perceptual_loss)

