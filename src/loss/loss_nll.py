from dataclasses import dataclass

from einops import rearrange
from jaxtyping import Float
from typing import Literal
from torch import Tensor, nn
from .loss import Loss


@dataclass
class LossNLLCfg:
    name: Literal['NLL']
    weight: float

@dataclass
class LossNLLCfgWrapper:
    nll: LossNLLCfg

class LossNLL(Loss[LossNLLCfg, LossNLLCfgWrapper]):
    def forward(
        self,
        prediction: Float[Tensor, "batch seq logits"],
        ground_truth: Float[Tensor, "batch seq"],
    ) -> Float[Tensor, ""]:
        
        # define NLL criterion
        criterion = nn.NLLLoss()
        # compute loss on reshaped tensors
        prediction = rearrange(prediction, "b s l -> (b s) l")
        ground_truth = rearrange(ground_truth, "b s -> (b s)")
        ground_truth = ground_truth.to(device=prediction.device)
        return criterion(prediction, ground_truth)