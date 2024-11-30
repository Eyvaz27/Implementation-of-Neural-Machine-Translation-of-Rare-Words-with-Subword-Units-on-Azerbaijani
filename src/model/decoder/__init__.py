from typing import Optional

from .decoder import Decoder
from .decoder_lstmCell import DecoderLSTMCell, DecoderLSTMCellCfg
from .decoder_gruCell import Decoder, DecoderMLPCfg

DECODERS = {"lstm_cell": DecoderPerceptron, 
            "gru_cell": DecoderMLP}

DecoderCfg = DecoderPerceptronCfg | DecoderMLPCfg

def get_decoder(cfg: DecoderCfg
                ) -> Decoder:
    return DECODERS[cfg.name](cfg)