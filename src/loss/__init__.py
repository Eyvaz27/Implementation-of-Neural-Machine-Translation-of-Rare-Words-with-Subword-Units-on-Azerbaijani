from .loss import Loss
from .loss_cce import LossCCE, LossCCECfgWrapper
from .loss_nll import LossNLL, LossNLLCfgWrapper

LOSSES = {
    LossCCECfgWrapper: LossCCE,
    LossNLLCfgWrapper: LossNLL}

LossCfgWrapper = LossCCECfgWrapper | LossNLLCfgWrapper

def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]