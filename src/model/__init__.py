from typing import Optional

from .nsp import NSP
from .bahdanau import Bahdanau, BahdanauCfg

MODELS = {"bahdanau": Bahdanau}

ModelCfg = BahdanauCfg

def get_model(cfg: ModelCfg
                ) -> NSP:
    return MODELS[cfg.name](cfg)