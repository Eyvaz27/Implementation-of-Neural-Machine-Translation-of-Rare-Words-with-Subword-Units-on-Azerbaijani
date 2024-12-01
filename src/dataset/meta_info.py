from typing import Literal, Optional
from dataclasses import dataclass

Stage = Literal["train", "validation", "test"]

DATASET_PATH = "/workspaces/Implementation-of-Neural-Machine-Translation-of-Rare-Words-with-Subword-Units-on-Azerbaijani/data_source"

@dataclass
class TokenizerCfg:
    vocab_size: Optional[int]
    min_frequency: int
    ckpt_path: Optional[str]