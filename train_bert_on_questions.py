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

from constants import targets
from modeling import BertOnQuestions
from training import Trainer

def apply_tokenizer(tokenizer, texts: List[str], maxlen) -> np.array:
    tokens = np.zeros((len(texts), maxlen), dtype=np.long)
    for i, text in enumerate(texts):
        text = ' '.join(text.strip().split(' ')[:maxlen])
        text_tokens = tokenizer.encode(text, max_length=maxlen, add_special_tokens=True)[:maxlen]
        tokens[i, :len(text_tokens)] = text_tokens 
    return tokens

"""
example:
python train_bert_on_questions.py --do_apex
"""
@email_sender(recipient_emails=["olivier.st.amand.1@gmail.com"], sender_email="yellow.bhaji@gmail.com")
def main(**args):
    # data
    targets_question = [x for x in targets if x.startswith('question')]
    train_df = pd.read_csv(os.path.join(args['data_dir'], 'train.csv'))
    texts = train_df.question_body.values
    labels = train_df[targets_question].values.astype(np.float32)

    tokenizer = transformers.BertTokenizer.from_pretrained(os.path.join(args['model_dir'], 'bert-base-uncased'))
    tokens =  apply_tokenizer(tokenizer, texts, args['maxlen'])

    tr_ids = pd.read_csv(os.path.join(args['data_dir'], f"train_ids_fold_{args['fold']}.csv"))['ids'].values
    val_ids = pd.read_csv(os.path.join(args['data_dir'], f"valid_ids_fold_{args['fold']}.csv"))['ids'].values

    x_train = tokens[tr_ids]
    y_train = labels[tr_ids]

    x_valid = tokens[val_ids]
    y_valid = labels[val_ids]

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train, dtype=torch.long), 
        torch.tensor(y_train, dtype=torch.float32)
    )

    valid_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_valid, dtype=torch.long), 
        torch.tensor(y_valid, dtype=torch.float32)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['bs'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args['bs'], shuffle=False)
    loaders = {'train': train_loader, 'valid': valid_loader}

    device = torch.device(args['device'])

    # train head

    params = BertOnQuestions.default_params()
    params['fc_dp'] = 0.
    model = BertOnQuestions(len(targets_question), args['model_dir'], **params)
    model.train_head_only()
    model.to(device)

    if args['do_wandb']:
        wandb.watch(model)
    
    optimizer = optim.Adam(model.optimizer_grouped_parameters, lr=1e-2)

    if args['do_apex']:
        # TODO tru O2
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    trainer = Trainer(**args)
    trainer.train(model, loaders, optimizer, epochs=args['epochs1'])

    # train all layers

    #del model
    #gc.collect()
    #torch.cuda.empty_cache()
    
    #params = BertOnQuestions.default_params()
    #params['fc_dp'] = args['dp']
    model.pooled_dp.p = args['dp']
    #model = BertOnQuestions(len(targets_question), args['model_dir'], **params)

    ckpt = args['head_ckpt'] if args['head_ckpt'] is not None else '.tmp/best.pth'
    # TODO check if file exists
    #model.load_state_dict(torch.load(ckpt))
    #model.to(device)

    model.train_all()
    optimizer = transformers.AdamW(model.optimizer_grouped_parameters, lr=2e-5)

    #if args['do_apex']:
    #    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    trainer = Trainer(**args)
    trainer.train(model, loaders, optimizer, epochs=args['epochs2'], warmup=0.5, warmdown=0.5)

    # save trained model and features

    out_dir = os.path.join(args['model_dir'], f"{args['out_dir']}_fold_{args['fold']}")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    torch.save(model.state_dict(), os.path.join(out_dir, 'model_state_dict.pth'))
    torch.save(args, os.path.join(out_dir, 'training_args.bin'))

# example: python train_bert_on_questions.py --do_apex --do_wandb --maxlen 256 --bs 8 --dp 0.1 --fold 0 --out_dir test
# trained model will be saved to model/test_fold_0
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
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--dp", default=0.4, type=float)
    parser.add_argument("--maxlen", default=512, type=int)
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