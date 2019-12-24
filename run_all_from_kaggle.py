from multiprocessing import Process
from functools import partial

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

if __name__ == "__main__":
    main()
