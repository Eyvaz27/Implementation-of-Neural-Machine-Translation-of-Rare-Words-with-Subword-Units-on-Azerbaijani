from dataclasses import dataclass
from typing import Literal, Optional

import os
import torch
from einops import repeat
from jaxtyping import Float
from torch import Tensor, nn
from .decoder import Decoder
from .huffman_trees.hsoftmax_layer import HSoftmaxLayer

@dataclass
class DecoderGRUCellCfg:
    name: Literal["gru_cell"]
    input_size: Optional[int]
    hidden_size: int
    bias: bool
    output_len: Optional[int]
    vocab_size: Optional[int]
    huffman_tree_dir: Optional[str]

class DecoderGRUCell(Decoder[DecoderGRUCellCfg]):
    def __init__(self, cfg: DecoderGRUCellCfg) -> None:
        super().__init__(cfg)
        self.init_decoder()
        self.init_hidden_cell()
    
    def init_decoder(self):
        self.cell = nn.GRUCell(input_size=self.cfg.input_size, 
                                hidden_size=self.cfg.hidden_size, 
                                bias=self.cfg.bias)
        self.alignment_model = nn.Sequential(
            nn.Linear(in_features=self.cfg.hidden_size + self.cfg.input_size, # considering encoder features 
                      out_features=1, bias=True))
        self.context_agg = nn.Sequential(
            nn.Linear(in_features=self.cfg.input_size * 2, # considering previous decoder outputs 
                      out_features=self.cfg.input_size, bias=True))
        self.next_token_in = nn.Sequential(
            nn.Linear(in_features=self.cfg.hidden_size, 
                      out_features=self.cfg.input_size, bias=True))
        
        self.huffman_flag = False
        if os.path.isdir(str(self.cfg.huffman_tree_dir)):
            self.next_token_out = HSoftmaxLayer(vocab_size=self.cfg.vocab_size, attention_dim=self.cfg.input_size,
                                                huffman_tree_dir=self.cfg.huffman_tree_dir, num_workers=1)
            self.huffman_flag = True
        else:
            self.next_token_out = nn.Sequential(
                nn.Linear(in_features=self.cfg.input_size, 
                        out_features=self.cfg.vocab_size, bias=True), nn.LogSoftmax(dim=1))
        self.out_act = nn.ReLU()

    def init_hidden_cell(self):
        self.y0 = torch.nn.Parameter(torch.randn(1, self.cfg.input_size)) # requires_grad
        self.h0 = torch.nn.Parameter(torch.randn(1, self.cfg.hidden_size)) # requires_grad

    def forward(
        self,
        features: Float[Tensor, "batch seq dim"]
        ) -> Float[Tensor, "batch seq c"]:

        # initialize hidden and cell states
        N, seq, _ = features.shape
        hx = repeat(self.h0, "1 dim -> n dim", n=N)

        outputs = [repeat(self.y0, "1 dim -> n dim", n=N)]
        output_logits = []
        for i in range(self.cfg.output_len):
             # following equation 5&6 in https://arxiv.org/pdf/1409.0473
             # we will only compare hidden_state against input
             hx_seq = repeat(hx, "n dim -> n s dim", s=seq)
             attn_input = torch.concatenate((features, hx_seq), dim=-1)       # [N, seq, hidden_dim + feature_dim]
             alignment_weights = torch.exp(self.alignment_model(attn_input))  # [N, seq, 1]
             context = torch.sum(alignment_weights * features, dim=1)         # [N dim] -> summing over T_x
             #
             # Below we have modelled [y_{t-1}; context] combination first and then apply RNN
             context_plus_input = self.context_agg(torch.concatenate((context, outputs[-1]), dim=-1))
             hx = self.cell(context_plus_input, hx)  # running GRU cell over sequence
             #
             # we have utilized Linear+ReLu mapping from hidden to inputShape
             next_token_inShape = self.next_token_in(self.out_act(hx))
             next_token = self.next_token_out(self.out_act(next_token_inShape))
             if self.huffman_flag: next_token = next_token.squeeze(dim=1)
             output_logits.append(next_token)
             outputs.append(next_token_inShape)
        
        output_logits = torch.stack(output_logits, dim=1) # [batch, out_seq, input_size]
        return output_logits