from multiprocessing import Process
from functools import partial
import pickle
import os

import numpy as np

from train_bert_on_all import main as run_train

import pdb

def main():
    params = {
        'do_apex': True,
        'do_wandb': True,
        'bs': 4, 
        'fold': 0, 
        'accumulation_step': 2,
        'epochs': 5,
        'lr': 2e-5, 
        'out_dir': 'outputs/bert_on_all_lm'
        'dp': 0.,
        'bert_wd': 0.01,
        'max_len_q_b': 150, 
        'model_dir': 'outputs/lm_finetuning'
        'warmup': 0.5, 
        'warmdown': 0.5
    }
    
    for fold_n in range(1):
        params['fold'] = fold_n
        p = Process(target=run_train, kwargs=params)
        p.start()
        p.join()

    pdb.set_trace()

    # get results from pickled files
    all_rho_vals = []
    for fold_n in range(5):
        with open(os.path.join(params['out_dir'], f"history_{fold_n}.pickle"), 'rb') as f:
            rho_vals = pickle.load(f)
            all_rho_vals.append(np.max(rho_vals))
    
    print(f"rho val: {np.mean(rho_vals):.4f} +- {np.std(rho_vals)}")
        
if __name__ == "__main__":
    main()
