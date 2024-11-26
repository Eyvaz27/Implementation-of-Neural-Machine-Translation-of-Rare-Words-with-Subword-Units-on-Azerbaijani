
import glob
import os
import os
import gc

os.environ["NCLL_BLOCKING_WAIT"] = "0"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
# os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from einops import rearrange

import time
import numpy as np
from .config import load_typed_root_config
from .loss import get_losses
from .loss.loss import Loss
from torchsummary import summary
from .dataset.data_module import DataModule
from .model.encoder import Encoder, get_encoder
from .model.decoder import Decoder, get_decoder
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class Trainer:
    #
    trainer_cfg: dict
    encoder: nn.Module
    decoder: Decoder
    losses: nn.ModuleList
    
    def __init__(
        self,
        trainer_cfg,
        encoder: Encoder,
        decoder: Decoder,
        losses: list[Loss], visdir, 
        datamodule) -> None:
        super().__init__()
        
        # trainer CFG is shared across nodes
        self.cfg = trainer_cfg
        self.vis_dir = visdir
        # Set up the model.
        self.encoder = encoder
        self.decoder = decoder
        self.losses = nn.ModuleList(losses)
        
        # # # setting up the dataloaders
        self.train_dataloader = datamodule.train_dataloader()
        self.validation_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()
        
        # # # setting up the logging info
        self.logging_dir = os.path.join(self.vis_dir, "tb_logging/")
        
        self.training_iter = 0
        self.ckpt_save_dict = {}
        self.validation_loss_best = torch.tensor([float("inf")])
        print(f"Start Trainer Class on CPU ...")
        
        self.model_optim_init()
        self.pretrained_ckpt_load()
    
    def pretrained_ckpt_load(self):
        # #
        # If possible, load checkpoint from current logging directory
        if os.path.isfile(os.path.join(self.vis_dir, "model_best.pth")):
            print('Loading an existing model from ', os.path.join(self.vis_dir, "model_best.pth"))
            saved_checkpoint = torch.load(os.path.join(self.vis_dir, "model_best.pth"),
                                          map_location="cuda:"+str(self.device_id_local))
            self.encoder.load_state_dict(saved_checkpoint["encoder_model_state_dict"])
            self.decoder.load_state_dict(saved_checkpoint["decoder_model_state_dict"])
            # # # 
            self.optimizer.load_state_dict(saved_checkpoint["optimizer_state_dict"])
            self.training_iter = saved_checkpoint["iteration"]
            self.validation_loss_best = saved_checkpoint["validation_loss_best"]
            print("\nSuccessfully loaded saved checkpoints and models\n")

        elif os.path.isfile(os.path.join(self.vis_dir, "model_latest.pth")):
            print('Loading an existing model from ', os.path.join(self.vis_dir, "model_latest.pth"))
            saved_checkpoint = torch.load(os.path.join(self.vis_dir, "model_latest.pth"),
                                          map_location="cuda:"+str(self.device_id_local)) 
            self.encoder.load_state_dict(saved_checkpoint["encoder_model_state_dict"])
            self.decoder.load_state_dict(saved_checkpoint["decoder_model_state_dict"])
            # # # 
            self.optimizer.load_state_dict(saved_checkpoint["optimizer_state_dict"])
            self.training_iter = saved_checkpoint["iteration"]
            self.validation_loss_best = saved_checkpoint["validation_loss_best"]
            print("\nSuccessfully loaded saved checkpoints and models\n")
            
        # Otherwise, try to load checkpoint from provided pretrained_ckpt directory
        elif self.cfg.pretrained_ckpt is not None:
            if os.path.isfile(os.path.join(self.cfg.pretrained_ckpt, "model_best.pth")):
                print('Loading an existing model from ', os.path.join(self.cfg.pretrained_ckpt, "model_best.pth"))
                saved_checkpoint = torch.load(os.path.join(self.cfg.pretrained_ckpt, "model_best.pth"),
                                            map_location="cuda:"+str(self.device_id_local)) 
                self.encoder.load_state_dict(saved_checkpoint["encoder_model_state_dict"])
                self.decoder.load_state_dict(saved_checkpoint["decoder_model_state_dict"])
                # # # 
                self.optimizer.load_state_dict(saved_checkpoint["optimizer_state_dict"])
                self.training_iter = saved_checkpoint["iteration"]
                self.validation_loss_best = saved_checkpoint["validation_loss_best"]
                print("\nSuccessfully loaded saved checkpoints and models\n")

            elif os.path.isfile(os.path.join(self.cfg.pretrained_ckpt, "model_latest.pth")):
                print('Loading an existing model from ', os.path.join(self.cfg.pretrained_ckpt, "model_latest.pth"))
                saved_checkpoint = torch.load(os.path.join(self.cfg.pretrained_ckpt, "model_latest.pth"),
                                            map_location="cuda:"+str(self.device_id_local)) 
                self.encoder.load_state_dict(saved_checkpoint["encoder_model_state_dict"])
                self.decoder.load_state_dict(saved_checkpoint["decoder_model_state_dict"])
                # # # 
                self.optimizer.load_state_dict(saved_checkpoint["optimizer_state_dict"])
                self.training_iter = saved_checkpoint["iteration"]
                self.validation_loss_best = saved_checkpoint["validation_loss_best"]
                print("\nSuccessfully loaded saved checkpoints and models\n")

            else: print("\n!!! NOT Possible to load saved checkpoints and models\n")
        else: print("\n!!! NOT Possible to load saved checkpoints and models\n")
    
    def model_optim_init(self): 
        
        self.encoder.train()
        self.decoder.train()
        
        # https://discuss.pytorch.org/t/optimizer-on-multi-neural-networks/20572?u=ptrblck
        self.optim_params = list(self.encoder.parameters()) + list(self.decoder.parameters())

        if self.cfg.optimizer.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.optim_params, lr=self.cfg.optimizer.base_lr, 
                                              eps=1e-15, betas=self.cfg.optimizer.betas)
        elif self.cfg.optimizer.optimizer_type == "adamw":
            self.optimizer = torch.optim.Adam(self.optim_params, lr=self.cfg.optimizer.base_lr,
                                              betas=self.cfg.optimizer.betas)
        elif self.cfg.optimizer.optimizer_type == "sgd":
            self.optimizer = torch.optim.Adam(self.optim_params, lr=self.cfg.optimizer.base_lr)
        # # 
        # Settint the sceduler for training
        if self.cfg.optimizer.scheduler=='cos':
            self.scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=self.cfg.epoch_num * len(self.train_dataloader), eta_min=self.cfg.optimizer.eta_min)
        elif self.cfg.optimizer.scheduler=='cos_ann':
            self.scheduler = CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=10000, T_mult=1, eta_min=self.cfg.optimizer.eta_min)
        else: 
            raise KeyError("Requisted Scheduler is not implemented yet ...")
        
    def training_step(self, batch):
        total_loss = 0.0
        
        features, gt_label = batch
        predictions = self.encoder(features)
        
        for loss_fn in self.losses:
            total_loss += loss_fn.forward(predictions, gt_label)
            
        return total_loss
    
    @torch.no_grad()
    def evaluation_step(self):
            
        total_eval_loss = 0.0
        for batch in self.validation_dataloader:
            total_loss = 0.0

            features, gt_label = batch
            predictions = self.encoder(features)
            
            for loss_fn in self.losses:
                total_loss += loss_fn.forward(predictions, gt_label)
            
            total_eval_loss += total_loss

        return total_eval_loss
                
    @torch.no_grad()
    def testing_step(self):
            
        total_test_loss = 0.0
        for batch in self.test_dataloader:
            total_loss = 0.0

            features, gt_label = batch
            predictions = self.encoder(features)
            
            for loss_fn in self.losses:
                total_loss += loss_fn.forward(predictions, gt_label)
            
            total_test_loss += total_loss
            
        return total_test_loss
    
    def train_model(self):
        # Start training model
        print("Model Training starts ...")
        for epoch_num in range(self.training_iter, self.cfg.epoch_num):

            start_time = time.time()
            epoch_loss_history = []
            for train_batch in self.train_dataloader:
                self.optimizer.zero_grad()
                training_loss = self.training_step(batch=train_batch)
                training_loss.backward()
                # Perform Gradient Clipping for stabilization
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 
                                            self.cfg.optimizer.gradient_clip_val)
                # Optimizer step
                self.optimizer.step()
                print(f"Training loss after {epoch_num} steps = {np.mean(epoch_loss_history)}")
                # # 
                # # Updating Scheduler
                if self.cfg.optimizer.scheduler != "none": # and new_scaler < old_scaler: 
                    self.scheduler.step()
                # # # logging loss values
                epoch_loss_history.append(training_loss.item())

            if (epoch_num + 1) % self.cfg.checkpointing.training_loss_log == 0: 
                print(f"Training loss after {epoch_num} steps = {np.mean(epoch_loss_history)}")
                print(f"Overall time required per epoch = {np.round(time.time() - start_time, decimals=3)} seconds")
            
            if (epoch_num + 1) % self.cfg.checkpointing.validation_log == 0:
                validation_loss = self.evaluation_step().detach().cpu()
                print(f"Validation loss after {epoch_num} steps = {validation_loss.item()}")
                if validation_loss < self.validation_loss_best:
                    self.validation_loss_best = validation_loss
                    self.ckpt_save_dict = {
                        "iteration": epoch_num,
                        "validation_loss_best": self.validation_loss_best,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "encoder_model_state_dict": self.encoder.state_dict(),
                        "decoder_model_state_dict": self.decoder.state_dict()}
                    torch.save(self.ckpt_save_dict, os.path.join(self.vis_dir, "model_best.pth"))
                    print("\nNew Best Model has been recorded ...\n")
                    
            if (epoch_num + 1) % self.cfg.checkpointing.testing_log == 0:
                testing_loss = self.testing_step().detach().cpu()
                print(f"Testing loss after {epoch_num} steps = {testing_loss.item()}")

            # #
            # # Saving Model Checkpoint
            if (epoch_num + 1) % self.cfg.checkpointing.checkpoint_iter == 0:
                latest_ckpt_save_dict = {
                    "iteration": epoch_num,
                    "validation_loss_best": self.validation_loss_best,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "encoder_model_state_dict": self.encoder.state_dict(),
                    "decoder_model_state_dict": self.decoder.state_dict()}
                torch.save(latest_ckpt_save_dict, os.path.join(self.vis_dir, "model_latest.pth"))



def model_training(cfg, vis_dir):
    
    cfg = load_typed_root_config(cfg)
    data_module = DataModule(cfg.dataset, cfg.data_loader, global_rank=0)

    cfg.model.encoder.input_size = cfg.dataset.embedding_dim
    encoder = get_encoder(cfg.model.encoder)

    cfg.model.decoder.input_size = encoder.feature_dim()
    cfg.model.decoder.output_size = cfg.dataset.class_count
    decoder = get_decoder(cfg.model.decoder)
    losses = get_losses(cfg.loss)

    trainer = Trainer(trainer_cfg=cfg.trainer, encoder=encoder, decoder=decoder, 
                      losses=losses, visdir=vis_dir, datamodule=data_module)
    trainer.train_model()

if __name__ == "__main__":
    # You need to set the ocnfigus directory properly
    experiments_out_dir = "/workspaces/NLP---Text-Classification-of-Coronavirus-Tweets/experiments/"
    newest_day_dir = max(glob.glob(os.path.join(experiments_out_dir, '*/')), key=os.path.getmtime)
    vis_dir = max(glob.glob(os.path.join(newest_day_dir, '*/')), key=os.path.getmtime)
    cfg_file_path = os.path.join(vis_dir, "configs.yaml")
    cfg = OmegaConf.load(cfg_file_path)
    model_training(cfg, vis_dir)