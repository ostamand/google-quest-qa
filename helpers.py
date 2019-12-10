import os

import random 
import numpy as np
import torch
from scipy.stats import spearmanr

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)

def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)

class EarlyStoppingSimple():
    """Early stopping based on a metric."""
    def __init__(self, model, patience=10, min_delta=1e-4):
        self._best_score = -np.inf
        self._best_step = -1
        self._training_done = False
        self._current_step = 0
        
        self.count = 0
        
        self.min_delta = min_delta
        self.model = model
        self.patience = patience
        
        if not os.path.exists('.tmp'):
            os.mkdir('.tmp')
            
    def step(self, value):
        self._current_step += 1
        if value > (self.best_score + self.min_delta):
            self.count = 0
            self._best_score = value
            self._best_step = self._current_step
            torch.save(self.model.state_dict(), '.tmp/best.pth')
        else:
            if self.count >= self.patience:
                self._training_done = True
            else: 
                self.count += 1
            
    def restore(self):
        print(f"Best epoch: {self.best_step:.6f}. Best score: {self.best_score:.6f}")
        print("Restoring...")
        self.model.load_state_dict(torch.load('.tmp/best.pth'))

    @property 
    def training_done(self):
        return self._training_done  
        
    @property 
    def best_score(self):
        return self._best_score
    
    @property 
    def best_step(self):
        return self._best_step