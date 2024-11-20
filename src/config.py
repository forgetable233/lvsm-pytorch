from dataclasses import dataclass
from typing import Literal, Optional, Type, TypeVar
from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from dataset.data_module import DataLoaderCfg
from dataset.dataset_re10k import DatasetRe10kCfg

T = TypeVar("T")

@dataclass
class RootCfg:
    dataset: DatasetRe10kCfg

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
