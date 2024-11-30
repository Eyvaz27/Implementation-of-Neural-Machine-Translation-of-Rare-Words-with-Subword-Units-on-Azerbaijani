from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from jaxtyping import Float
from torch import nn, Tensor

T = TypeVar("T")


class NSP(nn.Module, ABC, Generic[T]):
    cfg: T
    
    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg
    
    @abstractmethod
    def forward(
        self,
        input_tokens: Float[Tensor, "batch seqIn"]
        ) -> Float[Tensor, "batch seqOut vocab"]:
        pass