from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .loss import LossCfgWrapper
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg

@dataclass
class OptimizerCfg:
    optimizer_type: str
    base_lr: float
    betas: list[float]
    scheduler: str
    eta_min: float
    gradient_clip_val: float

@dataclass
class CheckpointingCfg:
    checkpoint_iter: int
    training_loss_log: int
    validate_iter: int
    test_iter: int

@dataclass
class ModelCfg:
    encoder: EncoderCfg
    decoder: DecoderCfg  

@dataclass
class TrainerCfg:
    epoch_num: int
    pretrained_ckpt: Optional[str]
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg

@dataclass
class RootCfg:
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    model: ModelCfg
    trainer: TrainerCfg
    loss: list[LossCfgWrapper]


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {list[LossCfgWrapper]: separate_loss_cfg_wrappers},
    )