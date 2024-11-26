
import hydra
import os
import os
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    vis_dir = os.getcwd()
    print(f"Using {vis_dir} for logging")
    dict_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True)
    print('Here are the general configs: \n', dict_cfg)
    # Saving the configuration file
    with open(os.path.join(vis_dir, "configs.yaml"), "w") as f: OmegaConf.save(cfg, f)
    print('Will use the following dir for saving details of the configs:', os.path.join(vis_dir, "configs.yaml"))
    logging_dir = os.path.join(vis_dir, "tb_logging/")
    print('Creating directory for logging checkpoints on', logging_dir)
    if(not os.path.exists(logging_dir)): os.makedirs(logging_dir)

if __name__ == "__main__": main()