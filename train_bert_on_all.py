import argparse
import os
import gc
from typing import List
import math
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from apex import amp
from knockknock import email_sender

try:
    import wandb
except:
    pass

from constants import targets
from modeling import BertOnQA_2
from training import Trainer

from datasets import DatasetQA

import pdb

#@email_sender(recipient_emails=["olivier.st.amand.1@gmail.com"], sender_email="yellow.bhaji@gmail.com")
def main(**args):
    # data
    train_df = pd.read_csv(os.path.join(args['data_dir'], 'train.csv'))
    
    if args['fold'] is not None:
        tr_ids = pd.read_csv(os.path.join(args['data_dir'],  f"train_ids_fold_{args['fold']}.csv"))['ids'].values
        val_ids = pd.read_csv(os.path.join(args['data_dir'], f"valid_ids_fold_{args['fold']}.csv"))['ids'].values
    else:
        # train on almost all the data
        tr_ids = np.arange(len(train_df))
        val_ids = None

    tokenizer = transformers.BertTokenizer.from_pretrained(args['model_dir'])

    train_dataset = DatasetQA(train_df, tokenizer, tr_ids, max_len_q_b=args['max_len_q_b'], max_len_q_t=30)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['bs'], shuffle=True)

    valid_loader = None
    if val_ids is not None:
        valid_dataset = DatasetQA(train_df, tokenizer, val_ids, max_len_q_b=args['max_len_q_b'], max_len_q_t=30)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args['bs'], shuffle=False)

    loaders = {'train': train_loader, 'valid': valid_loader}

    device = torch.device(args['device'])

    params = BertOnQA_2.default_params()
    params['fc_dp'] = args['dp']
    params['bert_wd'] = args['bert_wd']

    model = BertOnQA_2(len(targets), args['model_dir'], **params)
    model.to(device)

    if args['do_wandb']:
        wandb.init(project=args['project'], tags='bert_on_all')
        wandb.watch(model, log=None)

    optimizer = transformers.AdamW(model.optimizer_grouped_parameters, lr=args['lr'])

    if args['do_apex']:
        # TODO opt_level O2
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    # train all layers

    model.train_all()

    # TODO try: x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    trainer = Trainer(**args)
    rho_vals = trainer.train(model, loaders, optimizer, epochs=args['epochs'], warmup=args['warmup'], warmdown=args['warmdown'])

    # save trained model and features

    out_dir = args['out_dir']
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    torch.save(model.state_dict(), os.path.join(out_dir, f"model_state_dict_fold_{args['fold']}.pth"))
    torch.save(args, os.path.join(out_dir, f"training_args_fold_{args['fold']}.bin"))

    with open(os.path.join(args['out_dir'], f"history_{args['fold']}.pickle"), 'wb') as f:
        pickle.dump(rho_vals, f)


def get_default_params():
    return {
        'epochs': 5, 
        'bert_wd': 0.01,
        'lr': 2e-5, 
        'max_len_q_b': 150,
        'model_dir': 'outputs/lm_finetuning_all ',
        'out_dir': 'outputs/bert_on_all_lm',
        'data_dir': 'data',
        'fold': 0,
        'log_dir': '.logs',
        'seed': 42,
        'bs': 4,
        'dp': 0.1,
        'device': 'cuda',
        'do_apex': True,
        'do_wandb': True,
        'warmup': 0.5,
        'warmdown': 0.5,
        'clip': None,
        'accumulation_steps': 2,
        'project': 'google-quest-qa',
        'head_ckpt': None
    }

# python3  train_bert_on_all.py --do_apex --do_wandb --bs 4 --fold 0 --out_dir test_on_all --dp 0.1
# python3 train_bert_on_all.py --do_apex --do_wandb --bs 4 --fold 0 --out_dir outputs/test_on_all --dp 0.1 --bert_wd 0.01 --model_dir model/bert-base-uncased --warmup 0.5 --warmdown 0.5
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--bert_wd", default=0.0, type=float)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--max_len_q_b", default=150, type=int)
    parser.add_argument("--model_dir", default="model/bert-base-uncased-qa", type=str)
    parser.add_argument("--out_dir", default="outputs/bert-base-uncased-qa", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--log_dir", default=".logs", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--bs", default=8, type=int)
    parser.add_argument("--dp", default=0.1, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--do_apex", action='store_true')
    parser.add_argument("--do_wandb", action='store_true')
    parser.add_argument("--do_tb", action='store_true')
    parser.add_argument("--warmup", default=0.5, type=float)
    parser.add_argument("--warmdown", default=0.5, type=float)
    parser.add_argument("--clip", default=None, type=float)
    parser.add_argument("--accumulation_steps", default=2, type=int)
    parser.add_argument("--project", default="google-quest-qa", type=str)
    parser.add_argument("--head_ckpt", default=None, type=str)
    
    args = parser.parse_args()

    main(**args.__dict__)