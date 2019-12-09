import random 
import numpy as np
import torch
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)

class EarlyStoppingSimple():
    """Early stopping based on a metric."""
    def __init__(self, model, patience=10, min_delta=1e-4):
        self._best_score = np.inf
        self._best_step = -1
        self._training_done = False
        self._current_step = 0
        
        self.count = 0
        
        self.min_delta = min_delta
        self.model = model
        self.patience = patience
        
        if not os.path.exists('.temp'):
            os.mkdir('.temp')
            
    def step(self, value):
        self._current_step += 1
        if value < (self.best_score - self.min_delta):
            self.count = 0
            self._best_score = value
            self._best_step = self._current_step
            print()
            print('Saving checkpoint...')
            torch.save(self.model, '.temp/best.pth')
        else:
            if self.count >= self.patience:
                self._training_done = True
            else: 
                self.count += 1
            
    def restore(self):
        print(f"Best epoch: {self.best_step:.6f}. Best score: {self.best_score:.6f}")
        print("Restoring...")
        self.model = torch.load('.temp/best.pth')

    @property 
    def training_done(self):
        return self._training_done  
        
    @property 
    def best_score(self):
        return self._best_score
    
    @property 
    def best_step(self):
        return self._best_step