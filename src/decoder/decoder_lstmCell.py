from dataclasses import dataclass
from typing import Literal, Optional

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from .decoder import Decoder

@dataclass
class DecoderLSTMCellCfg:
    name: Literal["lstm_cell"]
    input_size: Optional[int]
    hidden_size: int
    bias: bool
    out_size: Optional[int]

class DecoderLSTMCell(Encoder[DecoderLSTMCellCfg]):
    def __init__(self, cfg: DecoderLSTMCellCfg) -> None:
        super().__init__(cfg)
        self.init_decoder()
    
    def init_decoder(self):
        self.cell = nn.LSTMCell(input_size=self.cfg.input_size, 
                                hidden_size=self.cfg.hidden_size, 
                                bias=self.cfg.bias)
        self.mapping = nn.Sequential(
            nn.Linear(in_features=self.cfg.hidden_size, 
                      out_features=self.cfg.out_size, bias=True),
            nn.ReLU())
        
    def reset_hidden_cell(self, N):
        h0 = torch.randn(N, self.cfg.hidden_size)
        c0 = torch.randn(N, self.cfg.hidden_size)
        return h0, c0

    def forward(
        self,
        features: Float[Tensor, "batch seq dim"]
        ) -> Float[Tensor, "batch seq c"]:

        N, _, _ = features.shape
        # # # initialize hidden and cell states
        h0, c0 = self.reset_hidden_cell(N)

        outputs = []
        for i in range(input.size()[0]):
             hx, cx = rnn(input[i], (hx, cx))
             output.append(hx)
        
        output = torch.stack(output, dim=0)

        # # # run LSTM model on token embeddings
        output, _ = self.model(features, (h0, c0))

        first_token = output[:, 0, :]
        last_token = output[:, -1, :]
        if self.cfg.aggregation == "sum":
            return first_token + last_token
        elif self.cfg.aggregation == "concat":
            return torch.concatenate([first_token, last_token], axis=-1)
        else:
            raise KeyError("Required aggregation is not implemented yet ...")
    
    def feature_dim(self):
        D = 2 if self.cfg.bidirectional else 1
        hidden_dim = D * self.cfg.hidden_size

        if self.cfg.aggregation == "sum":
            return hidden_dim
        elif self.cfg.aggregation == "concat":
            return 2 * hidden_dim