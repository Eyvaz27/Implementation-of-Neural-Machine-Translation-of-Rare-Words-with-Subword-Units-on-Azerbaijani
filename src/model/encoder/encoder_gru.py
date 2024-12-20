from dataclasses import dataclass
from typing import Literal, Optional

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from .encoder import Encoder

@dataclass
class EncoderGRUCfg:
    name: Literal["gru"]
    input_size: Optional[int]
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool

class EncoderGRU(Encoder[EncoderGRUCfg]):
    def __init__(self, cfg: EncoderGRUCfg) -> None:
        super().__init__(cfg)
        self.init_encoder()
    
    def init_encoder(self):
        self.model = nn.GRU(input_size=self.cfg.input_size, hidden_size=self.cfg.hidden_size, 
                            num_layers=self.cfg.num_layers, bias=self.cfg.bias, 
                            batch_first=self.cfg.batch_first, dropout=self.cfg.dropout,
                            bidirectional=self.cfg.bidirectional)
    def reset_hidden_state(self, N):
        D = 2 if self.cfg.bidirectional else 1
        if self.cfg.batch_first:
            h0 = torch.randn(D*self.cfg.num_layers, N, self.cfg.hidden_size)
        else:
            h0 = torch.randn(N, D*self.cfg.num_layers, self.cfg.hidden_size)
        return h0
    
    def forward(
        self,
        features: Float[Tensor, "batch seq dim"]
        ) -> Float[Tensor, "batch"]:
        
        N, _, _ = features.shape
        # # # initialize hidden and cell states
        h0 = self.reset_hidden_state(N)
        # # # run GRU model on token embeddings
        output, _ = self.model(features, h0)

        return output # [batch, seq, 2∗hidden_dim]
    
    @property
    def feature_dim(self):
        D = 2 if self.cfg.bidirectional else 1
        return D * self.cfg.hidden_size