import os
import gc
import argparse
import re
import pickle
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import transformers
from sklearn.preprocessing import OneHotEncoder

from train_bert_on_all import DatasetQA
from modeling import BertOnQA
from constants import targets

import pdb

"""
Requirements:

model/bert-base-uncased
outputs/bert_on_all

"""

def get_model(input_shape):
    pass

class MixModelDataset(torch.utils.data.Dataset):

    def __init__(self, model_dir, ckpt_dir, df, fold_n, enc=None, device='cuda'):
        super(MixModelDataset, self).__init__()
        self.df, self.fold_n = df, fold_n
        self.enc = enc
        self.model_dir, self.ckpt_dir = model_dir, ckpt_dir

        self.device = torch.device(device)

        self._get_data()

    def _get_data(self):
        self.data = {}
        # QA data 

        tokenizer = transformers.BertTokenizer.from_pretrained(os.path.join(self.model_dir, 'bert-base-uncased'))
        dataset = DatasetQA(self.df, tokenizer, max_len_q_b=150, max_len_q_t=30)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

        params = torch.load(os.path.join(self.ckpt_dir, f'training_args_fold_{self.fold_n}.bin'))
        params.pop('model_dir')

        model = BertOnQA(len(targets), os.path.join(self.model_dir, 'bert-base-uncased'), **params)
        model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, f'model_state_dict_fold_{self.fold_n}.pth')))
        model.eval()
        model.to(self.device)
        
        qa_data = []
        for batch in tqdm(loader, total=len(loader)):
            tokens, token_types, _ = batch
            with torch.no_grad():
                outs = model(tokens.to(self.device), attention_mask=(tokens > 0).to(self.device), token_type_ids=token_types.to(self.device))
                qa_data.append(outs.detach().cpu().numpy())
        
        self.qa_data = np.vstack(qa_data)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Category data

        find = re.compile(r"^[^.]*")

        self.df['netloc'] = self.df['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])
        
        if self.enc is None:
            self.enc = OneHotEncoder(handle_unknown='ignore')
            self.enc.fit(self.df[['category', 'netloc']].values)

        self.category = self.enc.transform(self.df[['category', 'netloc']].values).toarray()

    def __len__(self):
        return len(df)

    def __getitem__(self, idx):
        return self.qa_data[idx], self.category[idx]

def do_training(model, loaders, epochs):
    valid_preds = []
    for epoch_i in range(epochs):
        pass


def main(params):
    p = params 

    train_df = pd.read_csv(os.path.join(params['data_dir'], 'train.csv'))
    test_df = pd.read_csv(os.path.join(params['data_dir'], 'test.csv'))

    all_test_preds = []

    for fold_n in range(5):
        tr_ids = pd.read_csv(os.path.join(p['data_dir'],  f"train_ids_fold_{fold_n}.csv"))['ids'].values[:100] # TODO remove for dev
        val_ids = pd.read_csv(os.path.join(p['data_dir'], f"valid_ids_fold_{fold_n}.csv"))['ids'].values[:100]

        train_dataset = MixModelDataset(p['model_dir'], p['ckpt_dir'], train_df.iloc[tr_ids].copy(), fold_n)
        valid_dataset = MixModelDataset(p['model_dir'], p['ckpt_dir'], train_df.iloc[val_ids].copy(), fold_n, enc=train_dataset.enc)
        test_dataset = MixModelDataset(p['model_dir'], p['ckpt_dir'], test_df.copy(), fold_n, enc=train_dataset.enc)

        pdb.set_trace()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=p['bs'], shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=p['bs'], shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=p['bs'], shuffle=False)            
        loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

        # model = get_model(next(iter(train_loader)[0].shape[0]))

        # do training 
        test_preds = do_training(model, loaders, p['epochs'])
        all_test_preds.append(test_preds)

    return test_preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--model_dir", default="model", type=str)
    parser.add_argument("--ckpt_dir", default="outputs/bert_on_all", type=str)

    args = parser.parse_args()

    main(args.__dict__)










