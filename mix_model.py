import os
import gc
import argparse
import re
import pickle
from urllib.parse import urlparse
import os 
import pickle
from multiprocessing import Process, Manager
from functools import partial

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import transformers
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import rankdata

from datasets import DatasetQA
from modeling import BertOnQA_2
from constants import targets
from helpers_torch import set_seed
from helpers import compute_spearmanr, EarlyStoppingSimple
from schedulers import LearningRateWithUpDown
from sentence_encodings import get_use_features

try: 
    import wandb
except:
    pass

import pdb

"""
Requirements:

model/bert-base-uncased
outputs/bert_on_all
.tmp
"""

class MixModel(nn.Module):

    def __init__(self, qa_fc_size, qa_pool_size, use_embeds_size, use_dist_size, category_size):
        super(MixModel, self).__init__()
        #n_features = qa_fc_size + qa_pool_size + category_size + use_dist_size + use_embeds_size
        #self.fc_dp = nn.Dropout(0.4)
        #self.fc = nn.Linear(n_features, len(targets))

        self.layer_use_embeds = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(use_embeds_size + use_dist_size, 512),
            nn.ReLU(inplace=True)
        ) 

        self.layer_pool = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(qa_fc_size + qa_pool_size, 512),
            nn.ReLU(inplace=True)
        )

        self.layer_cat = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(category_size, category_size),
            nn.ReLU(inplace=True)
        )

        self.layer_top = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(category_size + 1024, len(targets))
        )

    def forward(self, qa_fc, qa_pool, use_embed, use_dist, category):
        x_embeds = torch.cat([use_embed, use_dist], dim=1)
        x_embeds = self.layer_use_embeds(x_embeds)

        x_pool = torch.cat([qa_fc, qa_pool], dim=1)
        x_pool = self.layer_pool(x_pool)

        x_cat = self.layer_cat(category)

        x = torch.cat([x_embeds, x_pool, x_cat], dim=1)

        out = self.layer_top(x)

        return out 

def run_qa_data(df, model_dir, ckpt_dir, fold_n):
    tokenizer = transformers.BertTokenizer.from_pretrained(model_dir)
    dataset = DatasetQA(df, tokenizer, max_len_q_b=150, max_len_q_t=30)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    params = torch.load(os.path.join(ckpt_dir, f'training_args_fold_{fold_n}.bin'))
    params.pop('model_dir')

    device = torch.device('cuda')
    model = BertOnQA_2(len(targets), model_dir, **params)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, f'model_state_dict_fold_{fold_n}.pth')))
    model.to(device)

    qa_poolings = []
    def extract_poolings(self, input, output):
        qa_poolings.append(output.detach().squeeze().cpu().numpy())

    h = model.pooling.register_forward_hook(extract_poolings)
        
    model.eval()
    qa_fc = []
    for batch in tqdm(loader, total=len(loader)):
        tokens, token_types, _ = batch
        with torch.no_grad():
            outs = model(tokens.to(device), attention_mask=(tokens > 0).to(device), token_type_ids=token_types.to(device))
            qa_fc.append(outs.detach().cpu().numpy())
        
    qa_fc = np.vstack(qa_fc).astype(np.float32)
    qa_pool = np.vstack(qa_poolings).astype(np.float32)

    # cleanup
    h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # save to .tmp
    with open('.tmp/qa_data.pickle', 'wb') as f:
         pickle.dump((qa_fc, qa_pool), f)

class MixModelDataset(torch.utils.data.Dataset):

    def __init__(self, model_dir, ckpt_dir, use_dir, df, fold_n, enc=None, cache_file=None, do_cache=False, device='cuda'):
        super(MixModelDataset, self).__init__()
        self.df, self.fold_n = df, fold_n
        self.enc = enc
        self.model_dir, self.ckpt_dir, self.cache_file = model_dir, ckpt_dir, cache_file
        self.use_dir = use_dir
        self.do_cache = do_cache

        self.device = torch.device(device)

        self._get_data()

    def _get_data(self):
        if targets[0] in self.df.columns:
            self.labels = self.df[targets].values.astype(np.float32)
        else: 
            self.labels = None

        # 1. QA data 
        path_cache_file = os.path.join('.tmp', self.cache_file) if self.cache_file is not None else None

        if self.do_cache and self.cache_file is not None:
            if os.path.exists(path_cache_file):
                with open(path_cache_file, 'rb') as f:
                    self.labels, self.qa_fc, self.qa_pool, self.use_embeds, self.use_dist, self.category = pickle.load(f)
                return

        # parallel process to make sure GPU memory is released
        p = Process(target=partial(run_qa_data, self.df, self.model_dir, self.ckpt_dir, self.fold_n))
        p.start()
        p.join() 

        # load from .tmp
        with open('.tmp/qa_data.pickle', 'rb') as f:
            self.qa_fc, self.qa_pool = pickle.load(f)

        # 2. Category data
        find = re.compile(r"^[^.]*")

        self.df['netloc'] = self.df['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

        if self.enc is None:
            self.enc = OneHotEncoder(handle_unknown='ignore')
            self.enc.fit(self.df[['category', 'netloc']].values)

        self.category = self.enc.transform(self.df[['category', 'netloc']].values).toarray().astype(np.float32)
        # in case we have some nans replace nan by 0
        self.category[np.isnan(self.category)] = 0 

        # 3. USE data
        self.use_embeds, self.use_dist = get_use_features(self.df, self.use_dir)

        if self.do_cache and self.cache_file is not None:
            with open(path_cache_file, 'wb') as f:
                pickle.dump((self.labels, self.qa_fc, self.qa_pool, self.use_embeds, self.use_dist, self.category), f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        labels = self.labels[idx] if self.labels is not None else []
        return self.qa_fc[idx], self.qa_pool[idx], self.use_embeds[idx], self.use_dist[idx], self.category[idx], labels

def do_evaluate(model, loader, with_labels=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lossf = nn.BCEWithLogitsLoss()

    model.eval()

    all_preds = []
    all_labels = []
    loss = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            qa_fc, qa_pool, use_embed, use_dist, category, labels = batch
            out = model(
                qa_fc.to(device), 
                qa_pool.to(device),
                use_embed.to(device),
                use_dist.to(device),
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
        wandb.init(project='google-quest-qa', tags=["mix_model"])
        wandb.watch(model, log=None)

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

            qa_fc, qa_pool, use_embed, use_dist, category, labels = batch

            bs = qa_fc.shape[0]

            outs = model(
                qa_fc.to(device), 
                qa_pool.to(device),
                use_embed.to(device),
                use_dist.to(device),
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
                wandb.log(logs, step=it)

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
                wandb.log(logs)

            val_rhos.append(rho_val)

            print(f"rho: {rho_val:.4f} (val), loss: {loss_val:.4f} (val)")

        if loaders['test']:
            test_preds = do_evaluate(model, loaders['test'], with_labels=False)
            all_test_preds.append(test_preds)
        
    if loaders['valid']:
        early_stopping.restore()

    # TODO return average over all epochs
    return test_preds, valid_preds, np.max(val_rhos)

def train_loop(train_df, test_df, fold_n, params):
    p = params

    tr_ids = pd.read_csv(os.path.join(p['fold_dir'],  f"train_ids_fold_{fold_n}.csv"))['ids'] # TODO remove for dev
    val_ids = pd.read_csv(os.path.join(p['fold_dir'], f"valid_ids_fold_{fold_n}.csv"))['ids']

    train_dataset = MixModelDataset(p['model_dir'], p['ckpt_dir'], p['use_dir'], train_df.iloc[tr_ids].copy(), fold_n, cache_file=f"mix_train_fold_{fold_n}.pickle", do_cache=p['do_cache'])
    valid_dataset = MixModelDataset(p['model_dir'], p['ckpt_dir'], p['use_dir'], train_df.iloc[val_ids].copy(), fold_n, enc=train_dataset.enc, cache_file=f"mix_valid_fold_{fold_n}.pickle", do_cache=p['do_cache'])
    test_dataset = MixModelDataset(p['model_dir'], p['ckpt_dir'], p['use_dir'], test_df.copy(), fold_n, enc=train_dataset.enc, cache_file=f"mix_test_fold_{fold_n}.pickle", do_cache=p['do_cache'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=p['bs'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=p['bs'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=p['bs'], shuffle=False)            
    loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

    qa_fc, qa_pool, use_embed, use_dist, cat, _ = next(iter(train_loader))
    model = MixModel(qa_fc.shape[1], qa_pool.shape[1], use_embed.shape[1], use_dist.shape[1], cat.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), p['lr'])

    # do training 
    do_wandb = False if fold_n == 0 else False
    # assume last epoch is best for for valid_preds...
    test_preds, valid_preds, val_rho = do_training(model, loaders, optimizer, params, do_wandb=do_wandb)

    # dump results
    with open('.tmp/train_loop.pickle', 'wb') as f:
        pickle.dump((test_preds, valid_preds, val_rho), f)

    # save model 
    torch.save(model.state_dict(), os.path.join(p['out_dir'], f"model_state_dict_fold_{fold_n}.pth"))
            
    with open(os.path.join(p['out_dir'], f"enc_fold_{fold_n}.pickle"), 'wb') as f:
        pickle.dump(train_dataset.enc, f)

    # cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

def main(params):
    p = params 

    # check output dir
    if not os.path.exists(p['out_dir']):
        os.mkdir(p['out_dir'])

    train_df = pd.read_csv(os.path.join(params['data_dir'], 'train.csv'))
    test_df = pd.read_csv(os.path.join(params['data_dir'], 'test.csv'))
    sub_df = pd.read_csv(os.path.join(params['data_dir'], 'sample_submission.csv'))

    test_preds_per_fold = []
    valid_preds_per_fold = []
    val_rhos = []
    for fold_n in range(5):
        pr = Process(target=partial(train_loop, train_df, test_df, fold_n, params))
        pr.start()
        pr.join() 
                    
        with open('.tmp/train_loop.pickle', 'rb') as f:
            test_preds, valid_preds, val_rho = pickle.load(f)

        val_rhos.append(val_rho)
        test_preds_per_fold.append(test_preds)
        valid_preds_per_fold.append(valid_preds)

    #TODO model on questions...

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
        'bs': 32,
        'epochs': 100,
        'lr': 1e-4, 
        'seed': 42,
        'warmup': 0.5,
        'warmdown': 0.5,
        'data_dir': 'data',
        'fold_dir': 'data',
        'model_dir': 'model',
        'use_dir': 'model/universal-sentence-encoder-large-5',
        'ckpt_dir': 'outputs/bert_on_all_1', 
        'sub_type': 1, 
        'do_cache': False, 
        'out_dir': 'outputs/mix_model', 
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=16, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-4, type=float) # 1e-4
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--warmup", default=0.5, type=float)
    parser.add_argument("--warmdown", default=0.5, type=float)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--use_dir", default="model/universal-sentence-encoder-large-5", type=str)
    parser.add_argument("--out_dir", default="outputs/mix_model", type=str)
    parser.add_argument("--fold_dir", default="data", type=str)
    parser.add_argument("--model_dir", default="model", type=str)
    parser.add_argument("--ckpt_dir", default="outputs/bert_on_all_1", type=str)
    parser.add_argument("--sub_type", default=1, type=int)
    parser.add_argument("--do_cache", action='store_true')

    args = parser.parse_args()

    main(args.__dict__)










