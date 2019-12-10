import sys
sys.path.insert(0, 'lib/transformers')
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

from constants import targets
from modeling import BertOnQuestions
from training import Trainer

#TODO remove
import pdb

def apply_tokenizer(tokenizer, texts: List[str], maxlen) -> np.array:
    tokens = np.zeros((len(texts), maxlen), dtype=np.long)
    for i, text in enumerate(texts):
        text = ' '.join(text.strip().split(' ')[:maxlen])
        text_tokens = tokenizer.encode(text, max_length=maxlen, add_special_tokens=True)[:maxlen]
        tokens[i, :len(text_tokens)] = text_tokens 
    return tokens

def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--model_dir", default="model", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--log_dir", default=".logs", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--maxlen", default=512, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--do_apex", action='store_true')
    parser.add_argument("--warmup", default=0.5, type=float)
    parser.add_argument("--warmdown", default=0.5, type=float)
    parser.add_argument("--clip", default=5.0, type=float)
    parser.add_argument("--accumulation_steps", default=2, type=int)
    # parser.add_argument("--do_head", action="store_true")
    # parser.add_argument("--head_ckpt", default="", type=str)

    args = parser.parse_args()

    # data
    targets_question = [x for x in targets if x.startswith('question')]
    train_df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    texts = train_df.question_body.values
    labels = train_df[targets_question].values.astype(np.float32)

    tokenizer = transformers.BertTokenizer.from_pretrained(os.path.join(args.model_dir, 'bert-base-uncased'))
    tokens =  apply_tokenizer(tokenizer, texts, args.maxlen)

    x_train, x_valid, y_train, y_valid = train_test_split(tokens, labels, random_state=args.seed, test_size=0.1)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train, dtype=torch.long), 
        torch.tensor(y_train, dtype=torch.float32)
    )

    valid_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_valid, dtype=torch.long), 
        torch.tensor(y_valid, dtype=torch.float32)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.bs, shuffle=False)
    loaders = {'train': train_loader, 'valid': valid_loader}

    # train head
    params = BertOnQuestions.default_params()
    params['fc_dp'] = 0.
    model = BertOnQuestions(len(targets_question), args.model_dir, **params)

    model.train_head_only()
    optimizer = optim.Adam(model.optimizer_grouped_parameters, lr=1e-2)
    device = torch.device(args.device)
    model.to(device)
    if args.do_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    trainer = Trainer(**args.__dict__)
    trainer.train(model, loaders, optimizer, epochs=20)

    # train all layers
    # TODO
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()