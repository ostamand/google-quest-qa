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
from modeling import BertOnQA
from training import Trainer

from torch.utils.data import Dataset

import pdb

class DatasetQA(Dataset):

    # TODO variable maxlen. for now fixed at 512
    def __init__(self, df, tokenizer, ids=None, max_len_q_b=150, max_len_q_t=30):
        super(DatasetQA, self).__init__()

        #df = df.iloc[:10] # for dev
        
        df['q_b_tokens'] = df['question_body'].apply(lambda x: tokenizer.encode(x, max_length=512, add_special_tokens=False))
        df['q_t_tokens'] = df['question_title'].apply(lambda x: tokenizer.encode(x, max_length=512, add_special_tokens=False))
        df['a_tokens'] = df['answer'].apply(lambda x: tokenizer.encode(x, max_length=512, add_special_tokens=False))
        
        # [PAD]: 0
        # [ANSWER]: 1
        # [QUESTION_BODY]: 2
        def process(row, how=0):
            # token ids:    [CLS] [QUESTION_BODY] question body [ANSWER] answer [SEP] [PAD]
            # token types:  0    ...                            1 ...
            # TODO token types = 0 for the [PAD]?
            if how == 0:
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

            # token ids:    [CLS] question body [SEP] answer [SEP] [PAD]
            # token types:  0    ...             0    1 ...        0 
            if how == 1:
                tokens = [tokenizer.cls_token_id] + (512-1)*[tokenizer.pad_token_id] 

                len_q = np.min([max_len_q_b, len(row['q_b_tokens'])])

                if len(row['a_tokens']) >= 512-3-len_q:
                    # need to truncate the answer and possibly the question
                    question_trunc = row['q_b_tokens'][:len_q]
                    answer_trunc = row['a_tokens'][:512-3-len_q]
                else: 
                    # full answer and maximum question length
                    answer_trunc = row['a_tokens']
                    question_trunc = row['q_b_tokens'][:512-3-len(answer_trunc)]
                
                combined = question_trunc + [tokenizer.sep_token_id] + answer_trunc + [tokenizer.sep_token_id]
                tokens[1:1+len(combined)] = combined

                token_types = [0] * (len(question_trunc)+2) + (len(answer_trunc)+1) * [1] + (512 - len(answer_trunc) - len(question_trunc) - 3) * [0]

                return tokens, token_types

            # token ids:    [CLS] question title [SEP] question body [SEP] answer [SEP] [PAD]
            # token types:  0     0 ...          0     1 ...         1     1 ...  1     0 ...
            if how==2:
                tokens = [tokenizer.cls_token_id] + (512-1)*[tokenizer.pad_token_id] 

                len_q_b = np.min([max_len_q_b, len(row['q_b_tokens'])])
                len_q_t = np.min([max_len_q_t, len(row['q_t_tokens'])])

                len_q = len_q_b + len_q_t

                if len(row['a_tokens']) >= 512-4-len_q:
                    # need to truncate the answer and possibly the questions
                    question_title_trunc = row['q_t_tokens'][:len_q_t]
                    question_body_trunc =  row['q_b_tokens'][:len_q_b]
                    answer_trunc = row['a_tokens'][:512-4-len_q_t-len_q_b]
                else:
                    # full answer and maximum question length
                    answer_trunc = row['a_tokens']
                    
                    # try with full question body and truncated question title
                    if len(answer_trunc) + len(row['q_b_tokens']) + len_q_t >= 512-4:
                        pdb.set_trace()
                        question_body_trunc = row['q_b_tokens']
                        question_title_trunc = row['q_t_tokens'][:512-4-len(answer_trunc)-len(question_body_trunc)]
                    # full question body, question title and answer
                    else:
                        question_body_trunc = row['q_b_tokens']
                        question_title_trunc = row['q_t_tokens']

                combined = question_title_trunc + [tokenizer.sep_token_id] + question_body_trunc + [tokenizer.sep_token_id] + answer_trunc + [tokenizer.sep_token_id]
                tokens[1:1+len(combined)] = combined

                token_types = [0] * (len(question_title_trunc)+2) + (len(question_body_trunc)+len(answer_trunc)+2) * [1] + (512 - len(answer_trunc) - len(question_body_trunc) - len(question_title_trunc)  - 4) * [0]

                return tokens, token_types

        df['all'] = df.apply(lambda x: process(x, how=1), axis=1)

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
    tokenizer = transformers.BertTokenizer.from_pretrained(args['model_dir'])

    train_dataset = DatasetQA(train_df, tokenizer, tr_ids, max_len_q_b=150)

    valid_dataset = DatasetQA(train_df, tokenizer, val_ids, max_len_q_b=150)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['bs'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args['bs'], shuffle=False)
    loaders = {'train': train_loader, 'valid': valid_loader}

    device = torch.device(args['device'])

    params = BertOnQA.default_params()
    params['fc_dp'] = args['dp']
    params['bert_wd'] = args['bert_wd']

    model = BertOnQA(len(targets), args['model_dir'], **params)
    model.to(device)

    if args['do_wandb']:
        wandb.init(project=args['project'], tags='bert_on_all')
        wandb.watch(model, log=None)

    optimizer = transformers.AdamW(model.optimizer_grouped_parameters, lr=2e-5)

    if args['do_apex']:
        # TODO opt_level O2
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    # train all layers

    model.train_all()

    # TODO try: x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    trainer = Trainer(**args)
    trainer.train(model, loaders, optimizer, epochs=args['epochs2'], warmup=args['warmup'], warmdown=args['warmdown'])

    # save trained model and features

    out_dir = args['out_dir']
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    torch.save(model.state_dict(), os.path.join(out_dir, f"model_state_dict_fold_{args['fold']}.pth"))
    torch.save(args, os.path.join(out_dir, f"training_args_fold_{args['fold']}.bin"))

# python3  train_bert_on_all.py --do_apex --do_wandb --bs 4 --fold 0 --out_dir test_on_all --dp 0.1
# python3 train_bert_on_all.py --do_apex --do_wandb --bs 4 --fold 0 --out_dir outputs/test_on_all --dp 0.1 --bert_wd 0.01 --model_dir outputs/qa_finetuning_20_epochs
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs1", default=10, type=int)
    parser.add_argument("--epochs2", default=5, type=int)
    parser.add_argument("--bert_wd", default=0.0, type=float)
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