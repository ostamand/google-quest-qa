from multiprocessing import Process
from functools import partial
import pickle
import os

import numpy as np

from from_kaggle import main as run_train

def main():
    params = {
        'data_dir': 'data',
        'out_dir': 'outputs/keras_qa',
        'model_dir': 'model/bert_en_uncased_L-12_H-768_A-12',
        'fold': 0,
        'bs': 8,
        'dp': 0.1,
        'epochs': 5,
        'seed': 42,
        't_max_len': 30,
        'q_max_len': 239,
        'a_max_len': 239,
        'label_smoothing': 0.,
        'lr': 2e-5,
        'warmup': 0.5, 
        'warmdown': 0.5,
        'do_wandb': False,
   }
    
    for fold_n in range(5):
        params['fold'] = fold_n
        p = Process(target=run_train, kwargs=params)
        p.start()
        p.join()

    # get results from pickled files
    rho_vals = []
    for fold_n in range(5):
        with open(os.path.join(params['out_dir'], f"history_{fold_n}.pickle"), 'rb') as f:
            result = pickle.load(f)
            rho_vals.append(np.max(rho_vals))
    
    print(f"rho val: {np.mean(rho_vals):.4f} +- {np.std(rho_vals)}")
        
if __name__ == "__main__":
    main()
