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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb
import transformers
from sklearn.preprocessing import OneHotEncoder

from train_bert_on_all import DatasetQA
from modeling import BertOnQA
from constants import targets
from helpers_torch import set_seed
from helpers import compute_spearmanr, EarlyStoppingSimple
from schedulers import LearningRateWithUpDown

import pdb

"""
Requirements:

model/bert-base-uncased
outputs/bert_on_all

"""

class MixModel(nn.Module):

    def __init__(self, n_features):
        super(MixModel, self).__init__()
        self.fc = nn.Linear(n_features, len(targets))

    def forward(self, qa_fc, qa_avg_pool, category):
        x = torch.cat([qa_fc, qa_avg_pool, category], dim=1)
        out = self.fc(x)
        return out 

class MixModelDataset(torch.utils.data.Dataset):

    def __init__(self, model_dir, ckpt_dir, df, fold_n, enc=None, device='cuda'):
        super(MixModelDataset, self).__init__()
        self.df, self.fold_n = df, fold_n
        self.enc = enc
        self.model_dir, self.ckpt_dir = model_dir, ckpt_dir

        self.device = torch.device(device)

        self._get_data()

    def _get_data(self):
        # QA data 

        tokenizer = transformers.BertTokenizer.from_pretrained(os.path.join(self.model_dir, 'bert-base-uncased'))
        dataset = DatasetQA(self.df, tokenizer, max_len_q_b=150, max_len_q_t=30)
        self.labels = dataset.labels

        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

        params = torch.load(os.path.join(self.ckpt_dir, f'training_args_fold_{self.fold_n}.bin'))
        params.pop('model_dir')

        model = BertOnQA(len(targets), os.path.join(self.model_dir, 'bert-base-uncased'), **params)
        model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, f'model_state_dict_fold_{self.fold_n}.pth')))
        model.to(self.device)

        qa_avg_pool = []
        def extract_avg_poolings(self, input, output):
            qa_avg_pool.append(output.detach().squeeze().cpu().numpy())

        h = model.avg_pool.register_forward_hook(extract_avg_poolings)
        
        model.eval()
        qa_fc = []
        for batch in tqdm(loader, total=len(loader)):
            tokens, token_types, _ = batch
            with torch.no_grad():
                outs = model(tokens.to(self.device), attention_mask=(tokens > 0).to(self.device), token_type_ids=token_types.to(self.device))
                qa_fc.append(outs.detach().cpu().numpy())
        
        self.qa_fc = np.vstack(qa_fc).astype(np.float32)
        self.qa_avg_pool = np.vstack(qa_avg_pool).astype(np.float32)

        h.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Category data

        find = re.compile(r"^[^.]*")

        self.df['netloc'] = self.df['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

        if self.enc is None:
            self.enc = OneHotEncoder(handle_unknown='ignore')
            self.enc.fit(self.df[['category', 'netloc']].values)

        self.category = self.enc.transform(self.df[['category', 'netloc']].values).toarray().astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        labels = self.labels[idx] if self.labels is not None else []
        return self.qa_fc[idx], self.qa_avg_pool[idx], self.category[idx], labels

def do_evaluate(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    preds = []
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            qa_fc, qa_avg_pool, category, labels = batch
            out = model(
                qa_fc.to(device), 
                qa_avg_pool.to(device), 
                category.to(device)
            )
            preds.append(out.detach().cpu().numpy())
    model.train()
    return np.vstack(preds)


def do_training(model, loaders, optimizer, params):
    p = params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
    set_seed(p['seed'])

    lossf = nn.BCEWithLogitsLoss()
    optimizer.zero_grad()
    running_loss = None

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2, verbose=True, factor=0.1)
    early_stopping = EarlyStoppingSimple(model, patience=5, min_delta=0)  

    lr_scheduler = LearningRateWithUpDown(
        optimizer, 
        p['epochs'] * len(loaders['train']), 
        warmup=p['warmup'], 
        warmdown=p['warmdown'], 
        ini_lr=1e-6, 
        final_lr=1e-6
    )

    model.to(device)

    it = 1 # global steps

    valid_preds = []
    test_preds = []
    for epoch_i in range(p['epochs']):
        model.train()

        if early_stopping.training_done:
            print(f"Early stopping on epoch {epoch_i-1}")
            break

        pb = tqdm(enumerate(loaders['train']), total=len(loaders['train']), desc=f"Epoch: {epoch_i+1}/{p['epochs']}")

        for batch_i, batch in pb:
            # step
            lr_scheduler.step()

            qa_fc, qa_avg_pool, category, labels = batch

            bs = qa_fc.shape[0]

            outs = model(
                qa_fc.to(device), 
                qa_avg_pool.to(device), 
                category.to(device)
            )

            loss = lossf(outs, labels.to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log
            logs = {}
            logs['lr/train'] = optimizer.param_groups[0]['lr']
            logs['loss/train'] = loss.item() / bs

            if running_loss:
                running_loss = 0.98 * running_loss + 0.02 * loss.item() / bs
            else:
                running_loss = loss.item() / bs

            pb.set_postfix(loss = running_loss)

            it+=1
        
        if loaders['valid']:
            valid_preds = do_evaluate(model, loaders['valid'])
                
            #scheduler.step(metrics['spearmanr'])
            #early_stopping.step(metrics['spearmanr'])

            #logs['loss/valid'] = metrics['loss']
            #logs['spearmanr/valid'] = metrics['spearmanr']
                
            #print(f"rho: {metrics['spearmanr']:.4f} (val), loss: {metrics['loss']:.4f} (val)")

        if loaders['test']:
            test_preds = do_evaluate(model, loaders['test'])
        
    if loaders['valid']:
        early_stopping.restore()

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

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=p['bs'], shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=p['bs'], shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=p['bs'], shuffle=False)            
        loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

        qa_fc, qa_avg_pool, cat, _ = next(iter(train_loader))

        model = MixModel(qa_fc.shape[1] + qa_avg_pool.shape[1]+ cat.shape[1])

        optimizer = torch.optim.Adam(model.parameters(), 1e-3)

        # do training 
        test_preds = do_training(model, loaders, optimizer, params)

        all_test_preds.append(test_preds)

    return test_preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--warmup", default=0.5, type=float)
    parser.add_argument("--warmdown", default=0.5, type=float)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--model_dir", default="model", type=str)
    parser.add_argument("--ckpt_dir", default="outputs/bert_on_all", type=str)

    args = parser.parse_args()

    main(args.__dict__)










