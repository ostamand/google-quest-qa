import argparse
import os
import gc
from typing import List
from functools import partial
from multiprocessing import Process
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from apex import amp

try:
    import wandb
except:
    pass

import pdb

from constants import targets
from modeling import BertOnQA_2
from training import Trainer

targets_for_tr = [x for x in targets if x.startswith('question')]

# not used anymore
def apply_tokenizer(tokenizer, texts: List[str], maxlen) -> np.array:
    tokens = np.zeros((len(texts), maxlen), dtype=np.long)
    for i, text in enumerate(texts):
        text = ' '.join(text.strip().split(' ')[:maxlen])
        text_tokens = tokenizer.encode(text, max_length=maxlen, add_special_tokens=True)[:maxlen]
        tokens[i, :len(text_tokens)] = text_tokens 
    return tokens

def process_for_questions(tokenizer, row):
    t = tokenizer.encode(row['question_title'], max_length=512, add_special_tokens=False)
    q = tokenizer.encode(row['question_body'],  max_length=512, add_special_tokens=False)

    t_len = len(t)
    q_len = len(q)

    # [CLS] question title [SEP] question body [SEP] [PAD]

    tokens = [tokenizer.cls_token_id] + (512-1)*[tokenizer.pad_token_id] 

    question_title_trunc = t

    if t_len + q_len + 3 > 512:
        question_body_trunc = q[:512-t_len-3]
    else:
        question_body_trunc = q

    combined = question_title_trunc + [tokenizer.sep_token_id] + question_body_trunc + [tokenizer.sep_token_id] 

    tokens[1:1+len(combined)] = combined

    token_types = [0] * (len(question_title_trunc)+2) + (len(question_body_trunc)+1) * [1] + (512 - len(question_title_trunc) - len(question_body_trunc) - 3) * [0]

    return tokens, token_types

#@email_sender(recipient_emails=["olivier.st.amand.1@gmail.com"], sender_email="yellow.bhaji@gmail.com")
def main(**args):
    # data
    
    train_df = pd.read_csv(os.path.join(args['data_dir'], 'train.csv'))

    tokenizer = transformers.BertTokenizer.from_pretrained(args['model_dir'])
    
    #targets_for_tr = [x for x in targets if x.startswith('question')]
    train_df['all'] = train_df.apply(lambda x: process_for_questions(tokenizer, x), axis=1)

    tokens = np.stack(train_df['all'].apply(lambda x: x[0]).values).astype(np.long)
    token_types = np.stack(train_df['all'].apply(lambda x: x[1]).values).astype(np.long)

    labels = train_df[targets_for_tr].values.astype(np.float32)

    if args['fold'] is not None:
        tr_ids = pd.read_csv(os.path.join(args['data_dir'], f"train_ids_fold_{args['fold']}.csv"))['ids'].values
        val_ids = pd.read_csv(os.path.join(args['data_dir'], f"valid_ids_fold_{args['fold']}.csv"))['ids'].values
    else:
        # train on almost all the data
        tr_ids = np.arange(labels.shape[0])
        val_ids = None

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(tokens[tr_ids], dtype=torch.long), 
        torch.tensor(token_types[tr_ids], dtype=torch.long), 
        torch.tensor(labels[tr_ids], dtype=torch.float32)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['bs'], shuffle=True)

    valid_loader = None
    if val_ids is not None:
        x_valid = tokens[val_ids]
        y_valid = labels[val_ids]
        valid_dataset = torch.utils.data.TensorDataset(
            torch.tensor(tokens[val_ids], dtype=torch.long), 
            torch.tensor(token_types[val_ids], dtype=torch.long), 
            torch.tensor(labels[val_ids], dtype=torch.float32)
        )
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args['bs'], shuffle=False)
    
    loaders = {'train': train_loader, 'valid': valid_loader}

    device = torch.device(args['device'])

    model = BertOnQA_2(len(targets_for_tr), args['model_dir'], **BertOnQA_2.default_params())
    model.to(device)

    if args['do_wandb']:
        wandb.init(project=args['project'], tags=['questions'])
        wandb.watch(model)

    optimizer = transformers.AdamW(model.optimizer_grouped_parameters, lr=args['lr1'])

    if args['do_apex']:
        # TODO tru O2
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    # train head

    if args['do_head']:
        model.train_head_only()
        trainer = Trainer(**args)
        trainer.train(model, loaders, optimizer, epochs=args['epochs1'])

    # train all layers

    model.train_all()

    for param_group in optimizer.param_groups:
        param_group['lr'] = args['lr2']

    trainer = Trainer(**args)
    trainer.train(model, loaders, optimizer, epochs=args['epochs2'], warmup=0.5, warmdown=0.5)

    # save trained model and features

    out_dir = args['out_dir']
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    suffix = f"_fold_{args['fold']}" if args['fold'] is not None else ""

    torch.save(model.state_dict(), os.path.join(out_dir, f"model_state_dict{suffix}.pth"))

    torch.save(args, os.path.join(out_dir, f"training_args{suffix}.bin"))

def get_preds(df, ckpt_dir, fold_n, params):
    p = Process(target=_run_get_preds, args=[df, ckpt_dir, fold_n, params])
    p.start()
    p.join()
    with open('.tmp/questions_preds.pickle', 'rb') as f:
        preds = pickle.load(f)
    return preds

def _run_get_preds(df, ckpt_dir, fold_n, params):
    tokenizer = transformers.BertTokenizer.from_pretrained(params['model_dir'])
    df['all'] = df.apply(lambda x: process_for_questions(tokenizer, x), axis=1)
    tokens = np.stack(df['all'].apply(lambda x: x[0]).values).astype(np.long)
    token_types = np.stack(df['all'].apply(lambda x: x[1]).values).astype(np.long)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(tokens, dtype=torch.long), 
        torch.tensor(token_types, dtype=torch.long)
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=params['bs'], shuffle=False)

    device = torch.device(params['device'])

    model = BertOnQA_2(len(targets_for_tr), params['model_dir'], **BertOnQA_2.default_params())
    model.to(device)
    
    ckpt_path = os.path.join(ckpt_dir, f'model_state_dict_fold_{fold_n}.pth')
    model.load_state_dict(torch.load(ckpt_path)) 
    model.eval()
    all_preds = []
    for batch in loader:
        with torch.no_grad():
            tokens, token_types = batch
            preds = model(tokens.to(device), attention_mask=(tokens > 0).to(device), token_type_ids=token_types.to(device))
            all_preds.append(torch.sigmoid(preds).cpu().numpy())
    
    all_preds = np.vstack(all_preds)

    with open('.tmp/questions_preds.pickle', 'wb') as f:
        pickle.dump(all_preds, f)

# example: python train_bert_on_questions.py --do_apex --do_wandb --maxlen 256 --bs 8 --dp 0.1 --fold 0 --out_dir test
# trained model will be saved to model/test_fold_0
# python train_bert_on_questions.py --do_apex --do_wandb --maxlen 256 --bs 8 --dp 0.1 --out_dir test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs1", default=10, type=int)
    parser.add_argument("--epochs2", default=5, type=int)
    parser.add_argument("--lr1", default=1e-2, type=float)
    parser.add_argument("--lr2", default=2e-5, type=float)
    parser.add_argument("--model_dir", default="model/bert-base-uncased", type=str)
    parser.add_argument("--out_dir", default="outputs/bert_questions", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--log_dir", default=".logs", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--bs", default=8, type=int)
    parser.add_argument("--dp", default=0.4, type=float)
    parser.add_argument("--maxlen", default=512, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--do_apex", action='store_true')
    parser.add_argument("--do_wandb", action='store_true')
    parser.add_argument("--do_tb", action='store_true')
    parser.add_argument("--do_head", action='store_true')
    parser.add_argument("--warmup", default=0.5, type=float)
    parser.add_argument("--warmdown", default=0.5, type=float)
    parser.add_argument("--clip", default=None, type=float)
    parser.add_argument("--accumulation_steps", default=2, type=int)
    parser.add_argument("--project", default="google-quest-qa", type=str)
    parser.add_argument("--head_ckpt", default=None, type=str)
    
    args = parser.parse_args()

    main(**args.__dict__)