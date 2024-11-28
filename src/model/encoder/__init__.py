from typing import Optional

from .encoder import Encoder
from .encoder_lstm import EncoderLSTM, EncoderLSTMCfg
from .encoder_gru import EncoderGRU, EncoderGRUCfg

ENCODERS = {"lstm": EncoderLSTM, 
            "gru": EncoderGRU}
EncoderCfg = EncoderLSTMCfg | EncoderGRUCfg 

def get_encoder(cfg: EncoderCfg
                ) -> Encoder:
    return ENCODERS[cfg.name](cfg)