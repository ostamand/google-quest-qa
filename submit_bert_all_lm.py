import argparse
import os 

import pandas as pd
import torch
import transformers
import numpy as np

from datasets import DatasetQA
from modeling import BertOnQA_2
from constants import targets

def main(params):
    test_df = pd.read_csv(os.path.join(params['data_dir'], 'test.csv'))
    sub_df = pd.read_csv(os.path.join(params['data_dir'], 'sample_submission.csv'))
    
    tokenizer = transformers.BertTokenizer.from_pretrained(params['model_dir'])
    test_dataset = DatasetQA(test_df, tokenizer)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=params['bs'], shuffle=False)

    device = torch.device(params['device'])
    model = BertOnQA_2(len(targets), params['model_dir'], **BertOnQA_2.default_params())
    model.to(device)

    test_preds_per_fold = []

    n = 1 if params['single_ckpt'] else 5
    for i in range(n):
        test_preds = []

        if params['single_ckpt']:
            ckpt_name = os.path.join(params['ckpt_dir'], f"model_state_dict_fold_None.pth")
        else:
            ckpt_name = os.path.join(params['ckpt_dir'], f"model_state_dict_fold_{i}.pth")

        model.load_state_dict(torch.load(ckpt_name))
        model.eval()

        for batch_i, batch in enumerate(loader):
            with torch.no_grad():
                x_batch, token_type_batch, y_batch = batch
                token_type_batch = token_type_batch.to(device)
                outs = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), token_type_ids=token_type_batch)
                test_preds.append(torch.sigmoid(outs).cpu().numpy())

        test_preds = np.vstack(test_preds)
        test_preds_per_fold.append(test_preds)
 
    test_preds = np.mean(test_preds_per_fold, axis=0)
    sub_df.iloc[:, 1:] = test_preds
    sub_df.to_csv('submission.csv', index=False)

def get_default_params():
    return {
        'model_dir': "model/bert-base-uncased",
        'ckpt_dir': 'outputs/bert_on_all_lm',
        'data_dir': 'data',
        'bs': 4, 
        'device': 'cuda',
        'single_ckpt': False
    }

# python3 submit_bert_all_lm.py --single_ckpt --ckpt_dir outputs/bert_on_all_lm_2_no_folds 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="model/bert-base-uncased", type=str)
    parser.add_argument("--ckpt_dir", default="outputs/bert_on_all_lm", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--bs", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--single_ckpt", action='store_true')
    
    args = parser.parse_args()

    main(args.__dict__)
