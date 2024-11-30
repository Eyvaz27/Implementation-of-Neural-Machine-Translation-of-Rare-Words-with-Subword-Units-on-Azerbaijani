from typing import Optional

from .decoder import Decoder
from .decoder_lstmCell import DecoderLSTMCell, DecoderLSTMCellCfg
from .decoder_gruCell import DecoderGRUCell, DecoderGRUCellCfg

DECODERS = {"lstm_cell": DecoderLSTMCell, 
            "gru_cell": DecoderGRUCell}

DecoderCfg = DecoderLSTMCellCfg | DecoderGRUCellCfg

def get_decoder(cfg: DecoderCfg
                ) -> Decoder:
    return DECODERS[cfg.name](cfg)