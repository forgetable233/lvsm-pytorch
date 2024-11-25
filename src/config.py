import os
from dataclasses import dataclass
from typing import Literal, Optional, Type, TypeVar
from dacite import Config, from_dict

from omegaconf import DictConfig, OmegaConf

from dataset.dataset_re10k import DatasetRe10kCfg


T = TypeVar("T")

# configs for dataloader
@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None

@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg

# model config
@dataclass
class DecoderCfg:
    dim: int
    depth: int
    heads: int
    attn_dim_head: int
    use_rmsnorm: bool
    add_value_residual: bool
    ff_glu: bool
    ff_post_act_ln: bool
    attn_qk_norm: bool

@dataclass
class ModelCfg:
    patch_size: int
    max_input_images: int
    channels: int
    width: int
    height: int
    rand_input_image_embed: bool
    perceptual_loss_weight: float
    decoder_kwargs: DecoderCfg
    
    log: bool
    use_wandb: bool
    lr: float
    model_params: str

# train configs
@dataclass
class TrainerCfg:
    limit_train_batches: int
    check_val_every_n_epoch: int
    max_epochs: int
    precision: str
    accelerator: str
    gradient_clip_val: float

@dataclass
class TrainCfg:
    seed: float
    resume: bool
    ckpt: str
    lr: float
    trainer: TrainerCfg
    
# logger configs
@dataclass
class WandbCfg:
    project: str
    name: str

@dataclass
class LoggerCfg:
    log: bool
    output_dir: str
    name: str
    use_wandb: bool
    wandb: WandbCfg

# Root config
@dataclass
class RootCfg:
    dataset: DatasetRe10kCfg
    data_loader: DataLoaderCfg
    model: ModelCfg
    train: TrainCfg
    logger: LoggerCfg

def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
    )

def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg
    )
