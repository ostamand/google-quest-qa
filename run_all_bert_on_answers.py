from multiprocessing import Process
from functools import partial
import pickle
import os

import numpy as np

from train_bert_on_answers import main as run_train

def main():
    params = {
        'epochs1': 10,
        'epochs2': 5,
        'lr1': 1e-2,
        'lr2': 2e-5,
        'model_dir': 'outputs/lm_finetuning_all',
        'out_dir': 'outputs/bert_on_answers_1',
        'data_dir': 'data',
        'fold': 0,
        'log_dir': '.logs',
        'seed': 42,
        'bs': 4, 
        'dp': 0.,
        'maxlen': 512, 
        'device': 'cuda',
        'do_apex': True,
        'do_wandb': True,
        'do_tb': False,
        'do_head': False,
        'warmup': 0.5, 
        'warmdown': 0.5, 
        'clip': None,
        'accumulation_step': 2,
        'project': 'google-quest-qa',
        'head_ckpt': None
    }
    
    for fold_n in range(5):
        params['fold'] = fold_n
        p = Process(target=run_train, kwargs=params)
        p.start()
        p.join()

    # get results from pickled files
    all_rho_vals = []
    for fold_n in range(5):
        with open(os.path.join(params['out_dir'], f"history_{fold_n}.pickle"), 'rb') as f:
            rho_vals = pickle.load(f)
            all_rho_vals.append(np.max(rho_vals))
    
    print(all_rho_vals)
    print(f"rho val: {np.mean(all_rho_vals):.4f} +- {np.std(all_rho_vals)}")
        
if __name__ == "__main__":
    main()
