import sys
sys.path.insert(0, 'lib/transformers')
import argparse
import os

import transformers
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from constants import targets
from train_bert_on_questions import apply_tokenizer
from modeling import BertOnQuestions

features = []

def extract_pooled(self, input, output):
    #global features
    _, pooled = output
    features.append(pooled.detach().cpu().numpy())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default="model/bert_q_3956.pth", type=str, required=True)
    parser.add_argument("--out_file", default="features/train_q_3956.npy", type=str, required=True)
    parser.add_argument("--in_file", default="data/train.csv", type=str, required=True)
    parser.add_argument("--model_dir", default="model", type=str)
    parser.add_argument("--bs", default=8, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    args = parser.parse_args()

    targets_question = [x for x in targets if x.startswith('question')]
    df = pd.read_csv(args.in_file)
    texts = df.question_body.values

    tokenizer = transformers.BertTokenizer.from_pretrained(os.path.join(args.model_dir, 'bert-base-uncased'))
    tokens =  apply_tokenizer(tokenizer, texts, 256)

    # TODO params should have been saved along with checkpoint
    model = BertOnQuestions(len(targets_question), args.model_dir, **{"fc_wd": 0., "fc_dp": 0., "bert_wd": 0})
    model.load_state_dict(torch.load(args.ckpt_path))
    model.bert.register_forward_hook(extract_pooled)
    model.eval()

    device = torch.device(args.device)
    model.to(device)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(tokens, dtype=torch.long)
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=False, drop_last=False)

    for (i, x_batch) in tqdm(enumerate(loader), total=len(loader)):
        with torch.no_grad():
            _ = model(x_batch[0].to(device))

    global features
    features = np.vstack(features)
    np.save(args.out_file, features)

# train data: python3 get_hidden_states_questions.py --ckpt_path model/bert_q_3956.pth --out_file features/train_q_3956.npy --in_file data/train.csv 
# tests data: python3 get_hidden_states_questions.py --ckpt_path model/bert_q_3956.pth --out_file features/test_q_3956.npy --in_file data/test.csv 
if __name__ == '__main__':
    main()
