from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from torch import Tensor, nn
from jaxtyping import Float
T = TypeVar("T")


class Decoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        features: Float[Tensor, "batch seqIn dim"]
        ) -> Float[Tensor, "batch seqOut logs"]:
        pass