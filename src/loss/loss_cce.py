from dataclasses import dataclass

from jaxtyping import Float
from typing import Literal
from torch import Tensor, nn
from .loss import Loss


@dataclass
class LossCCECfg:
    name: Literal['CCE']
    weight: float

@dataclass
class LossCCECfgWrapper:
    cce: LossCCECfg

class LossCCE(Loss[LossCCECfg, LossCCECfgWrapper]):
    def forward(
        self,
        prediction: Float[Tensor, "batch logits"],
        ground_truth: Float[Tensor, "batch"],
    ) -> Float[Tensor, ""]:
        criterion = nn.CrossEntropyLoss()
        ground_truth = ground_truth.to(device=prediction.device)
        return criterion(prediction, ground_truth)