from dataclasses import dataclass
from typing import Literal, Optional

import torch
from einops import repeat
from jaxtyping import Float
from torch import Tensor, nn

from .nsp import NSP
from .encoder import get_encoder, EncoderCfg
from .decoder import get_decoder, DecoderCfg

@dataclass
class EmbeddingCfg:
    name: Literal["torch"]
    embedding_dim: int
    scale_grad_by_freq: bool

@dataclass
class BahdanauCfg:
    name: Literal["bahdanau"]
    encoder: EncoderCfg
    decoder: DecoderCfg
    embedding: EmbeddingCfg
    vocab_size: Optional[int]
    max_length: Optional[int]

class Bahdanau(NSP[BahdanauCfg]):
    def __init__(self, cfg: BahdanauCfg) -> None:
        super().__init__(cfg)
        
        # initializing Embedding Layer
        self.embedding_layer = nn.Embedding(num_embeddings=self.cfg.vocab_size, 
                                            embedding_dim=self.cfg.embedding.embedding_dim,
                                            scale_grad_by_freq=self.cfg.embedding.scale_grad_by_freq)

        # initializing Encoder
        self.cfg.encoder.input_size = self.cfg.embedding.embedding_dim
        self.encoder = get_encoder(self.cfg.encoder)

        # initializing Decoder
        self.cfg.decoder.input_size = self.encoder.feature_dim
        self.cfg.decoder.vocab_size = self.cfg.vocab_size
        self.cfg.decoder.output_len = self.cfg.max_length
        self.decoder = get_decoder(self.cfg.decoder)

    def forward(
        self,
        input_tokens: Float[Tensor, "batch seqIn"]
        ) -> Float[Tensor, "batch seq vocab_size"]:

        # Embedding input tokens according to LookUp table
        embedded_input = self.embedding_layer(input_tokens).squeeze(dim=-2)
        encoded_features = self.encoder.forward(embedded_input)
        decoded_logits = self.decoder.forward(encoded_features)
        return decoded_logits