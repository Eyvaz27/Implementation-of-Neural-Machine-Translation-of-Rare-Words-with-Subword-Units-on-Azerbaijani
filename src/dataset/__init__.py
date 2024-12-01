from torch.utils.data import Dataset
from .aze_nsp import AZE_NSP_Dataset, AZE_NSP_DatasetCfg
from .meta_info import Stage

DATASETS: dict[str, Dataset] = {
    "aze_nsp": AZE_NSP_Dataset}

DatasetCfg = AZE_NSP_DatasetCfg

def get_dataset(
    cfg: DatasetCfg,
    stage: Stage, seed: int) -> Dataset:
    return DATASETS[cfg.name](cfg, stage, seed)