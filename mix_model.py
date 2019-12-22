import os
import gc
import argparse
import re
import pickle
from urllib.parse import urlparse
import os 
import pickle

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
#import wandb
import transformers
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import rankdata

from datasets import DatasetQA
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
.tmp
"""

class MixModel(nn.Module):

    def __init__(self, qa_fc_size, qa_avg_pool_size, category_size):
        super(MixModel, self).__init__()
        n_features = qa_fc_size + qa_avg_pool_size + category_size
        self.fc_dp = nn.Dropout(0.3)
        self.fc = nn.Linear(n_features, len(targets))

    def forward(self, qa_fc, qa_avg_pool, category):
        #x = torch.cat([qa_fc, qa_avg_pool, category], dim=1)
        x = torch.cat([qa_fc, qa_avg_pool, category], dim=1)
        out = self.fc(self.fc_dp(x))
        return out 

class MixModelDataset(torch.utils.data.Dataset):

    def __init__(self, model_dir, ckpt_dir, df, fold_n, enc=None, cache_file=None, do_cache=False, device='cuda'):
        super(MixModelDataset, self).__init__()
        self.df, self.fold_n = df, fold_n
        self.enc = enc
        self.model_dir, self.ckpt_dir, self.cache_file = model_dir, ckpt_dir, cache_file
        self.do_cache = do_cache

        self.device = torch.device(device)

        self._get_data()

    def _get_data(self):
        # QA data 
        path_cache_file = os.path.join('.tmp', self.cache_file)

        if self.do_cache and self.cache_file is not None:
            if os.path.exists(path_cache_file):
                with open(path_cache_file, 'rb') as f:
                    self.labels, self.qa_fc, self.qa_avg_pool, self.category = pickle.load(f)
                return

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

        if self.do_cache and self.cache_file is not None:
            with open(path_cache_file, 'wb') as f:
                pickle.dump((self.labels, self.qa_fc, self.qa_avg_pool, self.category), f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        labels = self.labels[idx] if self.labels is not None else []
        return self.qa_fc[idx], self.qa_avg_pool[idx], self.category[idx], labels

def do_evaluate(model, loader, with_labels=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lossf = nn.BCEWithLogitsLoss()

    model.eval()

    all_preds = []
    all_labels = []
    loss = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            qa_fc, qa_avg_pool, category, labels = batch
            out = model(
                qa_fc.to(device), 
                qa_avg_pool.to(device),
                category.to(device)
            )
            all_preds.append(torch.sigmoid(out).detach().cpu().numpy())
            if with_labels:
                loss += lossf(out, labels.to(device)).item()
                all_labels.append(labels.numpy())

    model.train()

    all_preds = np.vstack(all_preds)

    if with_labels:
        all_labels = np.vstack(all_labels)
        return all_preds, (all_labels, loss / len(all_labels))

    return all_preds

def do_training(model, loaders, optimizer, params, do_wandb=False):
    p = params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
    set_seed(p['seed'])

    lossf = nn.BCEWithLogitsLoss()
    optimizer.zero_grad()
    running_loss = None

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, verbose=True, factor=0.1)
    early_stopping = EarlyStoppingSimple(model, patience=10, min_delta=0)  

    lr_scheduler = LearningRateWithUpDown(
        optimizer, 
        p['epochs'] * len(loaders['train']), 
        warmup=p['warmup'], 
        warmdown=p['warmdown'], 
        ini_lr=1e-6, 
        final_lr=1e-6
    )

    model.to(device)

    if do_wandb:
        pass
        #wandb.init(project='google-quest-qa', tags=["mix_model"])
        #wandb.watch(model, log=None)

    it = 1 # global steps

    all_valid_preds = []
    all_test_preds = []
    val_rhos = []
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
            
            if do_wandb:
                pass
                #wandb.log(logs, step=it)

            if running_loss:
                running_loss = 0.98 * running_loss + 0.02 * loss.item() / bs
            else:
                running_loss = loss.item() / bs

            pb.set_postfix(loss = running_loss)

            it+=1
        
        if loaders['valid']:
            valid_preds, (labels, loss_val) = do_evaluate(model, loaders['valid'], with_labels=True)
            all_valid_preds.append(valid_preds)
            rho_val = compute_spearmanr(labels, valid_preds)
            #rho_val_mean = compute_spearmanr(labels, np.mean(all_valid_preds,axis=0))

            scheduler.step(rho_val)
            early_stopping.step(rho_val)

            logs = {}
            logs['spearmanr/valid'] = rho_val
            logs['loss/valid'] = loss_val

            if do_wandb:
                pass
                #wandb.log(logs)

            val_rhos.append(rho_val)

            print(f"rho: {rho_val:.4f} (val), loss: {loss_val:.4f} (val)")

        if loaders['test']:
            test_preds = do_evaluate(model, loaders['test'], with_labels=False)
            all_test_preds.append(test_preds)
        
    if loaders['valid']:
        early_stopping.restore()

    # TODO return average over all epochs
    return test_preds, np.max(val_rhos)

def main(params):
    p = params 

    train_df = pd.read_csv(os.path.join(params['data_dir'], 'train.csv'))
    test_df = pd.read_csv(os.path.join(params['data_dir'], 'test.csv'))
    sub_df = pd.read_csv(os.path.join(params['data_dir'], 'sample_submission.csv'))

    test_preds_per_fold = []
    val_rhos = []

    for fold_n in range(5):
        tr_ids = pd.read_csv(os.path.join(p['fold_dir'],  f"train_ids_fold_{fold_n}.csv"))['ids'] # TODO remove for dev
        val_ids = pd.read_csv(os.path.join(p['fold_dir'], f"valid_ids_fold_{fold_n}.csv"))['ids']

        #TODO add cache file
        train_dataset = MixModelDataset(p['model_dir'], p['ckpt_dir'], train_df.iloc[tr_ids].copy(), fold_n, cache_file=f"mix_train_fold_{fold_n}.pickle", do_cache=p['do_cache'])
        valid_dataset = MixModelDataset(p['model_dir'], p['ckpt_dir'], train_df.iloc[val_ids].copy(), fold_n, enc=train_dataset.enc, cache_file=f"mix_valid_fold_{fold_n}.pickle", do_cache=p['do_cache'])
        test_dataset = MixModelDataset(p['model_dir'], p['ckpt_dir'], test_df.copy(), fold_n, enc=train_dataset.enc, cache_file=f"mix_test_fold_{fold_n}.pickle", do_cache=p['do_cache'])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=p['bs'], shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=p['bs'], shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=p['bs'], shuffle=False)            
        loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

        qa_fc, qa_avg_pool, cat, _ = next(iter(train_loader))

        model = MixModel(qa_fc.shape[1], qa_avg_pool.shape[1], cat.shape[1])

        optimizer = torch.optim.Adam(model.parameters(), p['lr'])

        # do training 
        do_wandb = False if fold_n == 0 else False
        test_preds, val_rho = do_training(model, loaders, optimizer, params, do_wandb=do_wandb)

        val_rhos.append(val_rho)
        test_preds_per_fold.append(test_preds)

        # cleanup

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # do submission

    print("Printing submission file...")
    if params['sub_type'] == 1:
        test_preds = np.mean(test_preds_per_fold, axis=0)
    elif params['sub_type'] == 2:
        test_preds = np.array([np.array([rankdata(c) for c in p.T]).T for p in test_preds_per_fold]).mean(axis=0)
        max_val = test_preds.max() + 1
        test_preds = test_preds/max_val + 1e-12

    sub_df.iloc[:, 1:] = test_preds
    sub_df.to_csv('submission.csv', index=False)

    print(val_rhos)
    print(f"rho val: {np.mean(val_rhos):.4f} += {np.std(val_rhos):.4f}")

    return test_preds, val_rhos

def get_default_params():
    return {
        'bs': 16,
        'epochs': 100,
        'lr': 1e-4, 
        'seed': 42,
        'warmup': 0.5,
        'warmdown': 0.5,
        'data_dir': 'data',
        'fold_dir': 'data',
        'model_dir': 'model',
        'ckpt_dir': 'outputs/bert_on_all', 
        'sub_type': 1, 
        'do_cache': False
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=16, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-4, type=int) # 1e-4
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--warmup", default=0.5, type=float)
    parser.add_argument("--warmdown", default=0.5, type=float)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--fold_dir", default="data", type=str)
    parser.add_argument("--model_dir", default="model", type=str)
    parser.add_argument("--ckpt_dir", default="outputs/bert_on_all", type=str)
    parser.add_argument("--sub_type", default=1, type=int)
    parser.add_argument("--do_cache", action='store_true')

    args = parser.parse_args()

    main(args.__dict__)










