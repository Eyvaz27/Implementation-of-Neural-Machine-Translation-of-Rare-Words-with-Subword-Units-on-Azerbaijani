from torch.utils.data import Dataset
from .aze_nsp import TweetDataset, TweetDatasetCfg
from .meta_info import Stage

DATASETS: dict[str, Dataset] = {
    "tweet": TweetDataset}

DatasetCfg = TweetDatasetCfg

def get_dataset(
    cfg: DatasetCfg,
    stage: Stage, seed: int) -> Dataset:
    return DATASETS[cfg.name](cfg, stage, seed)