import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Union, Optional, Literal
from .meta_info import Stage, DATASET_PATH
from datasets import load_from_disk

@dataclass
class AZE_NSP_DatasetCfg:
    name: Literal["aze_nsp"]
    max_length: int
    padding: str
    truncation: bool
    embedding_dim: int
    eval_set_ratio: float

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
        from sklearn.model_selection import train_test_split

        raw_data = aze_nsp_dataset["train"].to_pandas()
        X = raw_data["partial_text"].map(lambda text: text.lower())
        y = raw_data["gold_ending"].map(lambda text: text.lower())
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        if self.stage == "test":
            X_test.reset_index(inplace=True, drop=True)
            y_test.reset_index(inplace=True, drop=True)

            self.input_text_samples = X_test.map(lambda sentence: "<bos> " + sentence + " <eos>")
            self.output_text_samples = y_test.map(lambda sentence: "<bos> " + sentence + " <eos>")
            

        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1/0.9, random_state=42)
        X_train_val.reset_index(inplace=True, drop=True)
        X_train.reset_index(inplace=True, drop=True)
        
        X_val.reset_index(inplace=True, drop=True)
        y_train_val.reset_index(inplace=True, drop=True)
        y_train.reset_index(inplace=True, drop=True)
        
        y_val.reset_index(inplace=True, drop=True)

        
        else: raw_csv_file = pd.read_csv(TRAIN_FILE_PATH, encoding = 'latin1')
        
        AZE_NSP__lower = lambda AZE_NSP_: AZE_NSP_.lower()
        feature_texts = raw_csv_file["OriginalAZE_NSP_"].map(AZE_NSP__lower)
        label_texts = raw_csv_file["Sentiment"].map(CLASS_MAP)

        if self.stage == "test":
            self.feature_texts = feature_texts
            self.label_texts = label_texts
        else:
            np.random.seed(seed=self.seed)
            random_perm_indexes = np.random.permutation(len(feature_texts))
            train_sample_indexes = random_perm_indexes[:int((1.0 - self.eval_set_ratio) * len(random_perm_indexes))]
            validation_sample_indexes = random_perm_indexes[int((1.0 - self.eval_set_ratio) * len(random_perm_indexes)):]
            if self.stage == "train":
                self.feature_texts = feature_texts[train_sample_indexes]
                self.label_texts = label_texts[train_sample_indexes]
            elif self.stage == "validation":
                self.feature_texts = feature_texts[validation_sample_indexes]
                self.label_texts = label_texts[validation_sample_indexes]
        # # # below code is important 
        self.feature_texts.reset_index(inplace=True, drop=True)
        self.label_texts.reset_index(inplace=True, drop=True)
    
    def init_tokenizer(self):
        self.auto_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.tokenizer = lambda AZE_NSP_: self.auto_tokenizer(AZE_NSP_, padding=self.cfg.padding, truncation=self.cfg.truncation,
                                                           max_length=self.cfg.max_length, return_tensors="pt")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased", 
                                                    torch_dtype=torch.float32, # because network needs fp32
                                                    attn_implementation="sdpa").eval()
        self.embedding_layer = lambda AZE_NSP__tokens: self.bert_model.embeddings(AZE_NSP__tokens['input_ids'])
        
    def __len__(self):
        return len(self.feature_texts)
    
    def __getitem__(self, idx):
        # # # 
        # # # retrieving samples from the dataset
        sample_AZE_NSP_ = self.feature_texts[idx]
        sample_label = self.label_texts[idx]
        with torch.no_grad():
            token_embedding = self.embedding_layer(self.tokenizer(sample_AZE_NSP_)).squeeze(axis=0)
        return token_embedding, sample_label
    