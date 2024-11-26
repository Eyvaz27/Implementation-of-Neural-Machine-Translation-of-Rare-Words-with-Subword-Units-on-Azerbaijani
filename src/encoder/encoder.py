from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from torch import Tensor, nn
from jaxtyping import Float
T = TypeVar("T")


class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        features: Float[Tensor, "batch seq dim"]
        ) -> Float[Tensor, "batch"]:
        pass

    @property
    def feature_dim(self): 
        pass