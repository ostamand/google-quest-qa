import os
import transformers
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc

from constants import targets
from train_bert_on_questions import apply_tokenizer
from modeling import BertOnQuestions

features = []

def extract_pooled(self, input, output):
    global features
    _, pooled = output
    features.append(pooled.detach().cpu().numpy())

def run(df, model_dir, ckpt_path, bs=8, device='cuda'):
    global features
    features = []

    targets_question = [x for x in targets if x.startswith('question')]
    texts = df.question_body.values

    tokenizer = transformers.BertTokenizer.from_pretrained(os.path.join(model_dir, 'bert-base-uncased'))
    tokens = apply_tokenizer(tokenizer, texts, 256)

    # TODO params should have been saved along with checkpoint
    model = BertOnQuestions(len(targets_question), model_dir, **{"fc_wd": 0., "fc_dp": 0., "bert_wd": 0})
    model.load_state_dict(torch.load(ckpt_path))
    model.bert.register_forward_hook(extract_pooled)
    model.eval()

    device = torch.device(device)
    model.to(device)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(tokens, dtype=torch.long)
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False)

    for (i, x_batch) in tqdm(enumerate(loader), total=len(loader)):
        with torch.no_grad():
            _ = model(x_batch[0].to(device))

    features = np.vstack(features)

    del model 
    gc.collect()
    torch.cuda.empty_cache()

    return features