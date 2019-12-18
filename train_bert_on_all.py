import argparse
import os
import gc
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from apex import amp
from knockknock import email_sender
import wandb

from constants import targets
from modeling import BertOnQuestions
from training import Trainer

from torch.utils.data import Dataset

import pdb

class DatasetQA(Dataset):

    # TODO variable maxlen. for now fixed at 512
    def __init__(self, df, tokenizer, ids=None, max_len_q_b=150):
        super(DatasetQA, self).__init__()
        
        # TODO check if padding is before SEP or not...
        df['q_b_tokens'] = df['question_body'].apply(lambda x: tokenizer.encode(x, max_length=512, add_special_tokens=False))
        df['a_tokens'] = df['answer'].apply(lambda x: tokenizer.encode(x, max_length=512, add_special_tokens=False))
        
        # [CLS] [QUESTION_BODY] ... [ANSWER] ... [SEP] ... [PAD] ... [PAD] 
        # [PAD]: 0
        # [ANSWER]: 1
        # [QUESTION_BODY]: 2
        def process(row):
            tokens = [tokenizer.cls_token_id] + [2] + (512-2)*[tokenizer.pad_token_id] 
        
            len_q = np.min([max_len_q_b, len(row['q_b_tokens'])])

            if len(row['a_tokens']) >= 512-4-len_q:
                # need to truncate the answer and possibly the question
                question_trunc = row['q_b_tokens'][:len_q]
                answer_trunc = row['a_tokens'][:512-4-len_q]
            else: 
                # full answer and maximum question length
                answer_trunc = row['a_tokens']
                question_trunc = row['q_b_tokens'][:512-4-len(answer_trunc)]
        
            combined = question_trunc + [1] + answer_trunc + [tokenizer.sep_token_id]
            tokens[2:2+len(combined)] = combined

            len_q += 2 # to consider special tokens
            token_types = [0] * len_q + (512-len_q) * [1]

            return tokens, token_types

        df['all'] = df.apply(lambda x: process(x), axis=1)

        self.labels = df[targets].values.astype(np.float32)
        self.tokens = np.stack(df['all'].apply(lambda x: x[0]).values).astype(np.long)
        self.token_types = np.stack(df['all'].apply(lambda x: x[1]).values).astype(np.long)

        if ids is not None:
            self.labels = self.labels[ids]
            self.tokens = self.tokens[ids]
            self.token_types = self.token_types[ids]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.tokens[idx], self.token_types[idx], self.labels[idx]

#@email_sender(recipient_emails=["olivier.st.amand.1@gmail.com"], sender_email="yellow.bhaji@gmail.com")
def main(**args):
    # data
    train_df = pd.read_csv(os.path.join(args['data_dir'], 'train.csv'))
    
    if args['fold'] is not None:
        tr_ids = pd.read_csv(os.path.join(args['data_dir'],  f"train_ids_fold_{args['fold']}.csv"))['ids'].values
        val_ids = pd.read_csv(os.path.join(args['data_dir'], f"valid_ids_fold_{args['fold']}.csv"))['ids'].values
    else:
        # train on almost all the data
        tr_ids, val_ids = train_test_split(np.arange(len(train_df)), test_size=0.05, random_state=args['seed'])

    # TODO change for bert-base-uncased-qa. ie finetuned LM 
    tokenizer = transformers.BertTokenizer.from_pretrained(os.path.join(args['model_dir'], 'bert-base-uncased'))

    train_dataset = DatasetQA(train_df, tokenizer, tr_ids, max_len_q_b=150)

    valid_dataset = DatasetQA(train_df, tokenizer, val_ids, max_len_q_b=150)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['bs'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args['bs'], shuffle=False)
    loaders = {'train': train_loader, 'valid': valid_loader}

    device = torch.device(args['device'])

    # train head

    params = BertOnQuestions.default_params()
    params['fc_dp'] = 0.
    model = BertOnQuestions(len(targets), args['model_dir'], **params)
    model.train_head_only()
    model.to(device)

    if args['do_wandb']:
        wandb.init(project=args['project'], tags='bert_on_all')
        wandb.watch(model, log=None)
    
    optimizer = transformers.AdamW(model.optimizer_grouped_parameters, lr=1e-3)

    if args['do_apex']:
        # TODO opt_level O2
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    trainer = Trainer(**args)
    trainer.train(model, loaders, optimizer, epochs=args['epochs1'])

    # train all layers

    model.pooled_dp.p = args['dp']
    model.train_all()

    for param_group in optimizer.param_groups:
        param_group['lr'] = 2e-5

    trainer = Trainer(**args)
    trainer.train(model, loaders, optimizer, epochs=args['epochs2'], warmup=0.5, warmdown=0.5)

    # save trained model and features

    if args['fold']:
        out_dir = os.path.join(args['model_dir'], f"{args['out_dir']}_fold_{args['fold']}")
    else: 
        out_dir = os.path.join(args['model_dir'], f"{args['out_dir']}")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    torch.save(model.state_dict(), os.path.join(out_dir, 'model_state_dict.pth'))
    torch.save(args, os.path.join(out_dir, 'training_args.bin'))

# example: python3 train_bert_on_all.py --do_apex --do_wandb --bs 4 --fold 0 --out_dir test_on_all
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs1", default=10, type=int)
    parser.add_argument("--epochs2", default=5, type=int)
    parser.add_argument("--model_dir", default="model", type=str)
    parser.add_argument("--out_dir", default="bert_questions", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--log_dir", default=".logs", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--bs", default=8, type=int)
    parser.add_argument("--dp", default=0.4, type=float)
    parser.add_argument("--maxlen", default=256, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--do_apex", action='store_true')
    parser.add_argument("--do_wandb", action='store_true')
    parser.add_argument("--do_tb", action='store_true')
    parser.add_argument("--warmup", default=0.5, type=float)
    parser.add_argument("--warmdown", default=0.5, type=float)
    parser.add_argument("--clip", default=10.0, type=float)
    parser.add_argument("--accumulation_steps", default=2, type=int)
    parser.add_argument("--project", default="google-quest-qa", type=str)
    parser.add_argument("--head_ckpt", default=None, type=str)
    
    args = parser.parse_args()

    main(**args.__dict__)