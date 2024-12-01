import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Union, Optional, Literal
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from .meta_info import Stage, DATASET_PATH, TokenizerCfg
from sklearn.model_selection import train_test_split
from datasets import load_from_disk
from tokenizers import Tokenizer

@dataclass
class AZE_NSP_DatasetCfg:
    name: Literal["aze_nsp"]
    max_length: int
    padding: str
    truncation: str
    eval_set_ratio: float
    tokenizer: TokenizerCfg

class AZE_NSP_Dataset(Dataset):
    def __init__(self,
                 cfg: AZE_NSP_DatasetCfg, stage: Stage, 
                 random_seed: Optional[int]):
        
        self.eval_set_ratio = cfg.eval_set_ratio
        assert self.eval_set_ratio>=0 and self.eval_set_ratio<=1.0

        self.seed = random_seed if random_seed else 42
        self.stage = stage
        self.cfg = cfg

        # read data from memory
        self.load_text_samples()
        self.init_tokenizer()

    def load_text_samples(self):

        aze_nsp_dataset = load_from_disk(dataset_path=DATASET_PATH)
        raw_data = aze_nsp_dataset["train"].to_pandas()
        X = raw_data["partial_text"].map(lambda text: text.lower())
        y = raw_data["gold_ending"].map(lambda text: text.lower())
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=self.cfg.eval_set_ratio, random_state=42)

        if self.stage == "test":
            X_test.reset_index(inplace=True, drop=True)
            y_test.reset_index(inplace=True, drop=True)
            self.input_text_samples = X_test.map(lambda sentence: "<bos> " + sentence + " <eos>")
            self.output_text_samples = y_test.map(lambda sentence: "<bos> " + sentence + " <eos>")
        
        else:
            validation_ratio = self.cfg.eval_set_ratio / (1.0 - self.cfg.eval_set_ratio)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_ratio, random_state=42)
            X_train_val.reset_index(inplace=True, drop=True)
            X_train.reset_index(inplace=True, drop=True)
            X_val.reset_index(inplace=True, drop=True)
            y_train_val.reset_index(inplace=True, drop=True)
            y_train.reset_index(inplace=True, drop=True)
            y_val.reset_index(inplace=True, drop=True)

            if not os.path.isfile(str(self.cfg.tokenizer.ckpt_path)):
                self.full_input_set = X_train_val
            if self.stage == "train":
                self.input_text_samples = X_train.map(lambda sentence: "<bos> " + sentence + " <eos>")
                self.output_text_samples = y_train.map(lambda sentence: "<bos> " + sentence + " <eos>")
            else:
                self.input_text_samples = X_val.map(lambda sentence: "<bos> " + sentence + " <eos>")
                self.output_text_samples = y_val.map(lambda sentence: "<bos> " + sentence + " <eos>")
    
    def init_tokenizer(self):
        if os.path.isfile(str(self.cfg.tokenizer.ckpt_path)):
            self.tokenizer = Tokenizer.from_file(self.cfg.tokenizer.ckpt_path)
            self.cfg.tokenizer.vocab_size = self.tokenizer.get_vocab_size()
        else: self.fit_tokenizer()
        # self.tokenizer.enable_padding(direction=self.cfg.padding, length=self.cfg.max_length)
        # self.tokenizer.enable_truncation(max_length=self.cfg.max_length, 
        #                                  direction=self.cfg.truncation)

    def fit_tokenizer(self):
        # Initialize a tokenizer
        self.tokenizer = Tokenizer(models.BPE())
        # Customize pre-tokenization and decoding
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        # And then train
        initial_aze_alphabet = list(set(' '.join(self.full_input_set)))
        # defining parameters for BPE
        trainer = trainers.BpeTrainer(
            vocab_size=self.cfg.tokenizer.vocab_size, 
            min_frequency=self.cfg.tokenizer.min_frequency,
            initial_alphabet=initial_aze_alphabet,
            special_tokens=["<bos>", "<eos>", "<pad>", "<mask>"])

        # training tokenizer on a given Train/Val set
        self.tokenizer.train_from_iterator(self.full_input_set.map(lambda sentence: "<bos> " + 
                                                                   sentence + " <eos>"), trainer=trainer)
        self.tokenizer.save(self.cfg.tokenizer.ckpt_path, pretty=True)
        
    def __len__(self):
        return len(self.input_text_samples)
    
    def __getitem__(self, idx):
        # # # 
        # # # retrieving samples from the dataset
        # Last portion of the input sentence will be retrieved
        self.tokenizer.enable_padding(length=self.cfg.max_length, direction="left")
        self.tokenizer.enable_truncation(max_length=self.cfg.max_length, direction="left")
        input_context = self.tokenizer.encode(self.input_text_samples[idx]).ids

        # first portion of the output sentence will be retrieved
        self.tokenizer.enable_padding(length=self.cfg.max_length, direction="right")
        self.tokenizer.enable_truncation(max_length=self.cfg.max_length, direction="right")
        output_sentence = self.tokenizer.encode(self.output_text_samples[idx]).ids

        return {"input": torch.LongTensor(np.array(input_context).reshape(-1, 1)),
                "output": torch.LongTensor(np.array(output_sentence).reshape(-1, 1))}
    