import numpy as np

class LearningRateWithUpDown():
    """Add warmup & warmdown to the optimizer learning rate."""
    def __init__(self, optimizer, total_steps, warmup=0.1, warmdown=0.1, ini_lr=1e-6, final_lr=1e-6, min_lr=1e-6):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup = warmup
        self.warmdown = warmdown
        self.ini_lr = ini_lr
        self.final_lr = final_lr
        self.min_lr = min_lr
        
        self.base_lr = optimizer.param_groups[0]['lr']
        
        self._current_step = 0
    
    def step(self):
        current_lr = self.optimizer.param_groups[0]['lr']
        
        lr = None 
        
        # warmup
        if self.warmup > 0:
            warmup = int(self.total_steps * self.warmup)
            if self._current_step <= warmup: 
                lr = self._current_step / warmup * (self.base_lr - self.ini_lr) + self.ini_lr
        
        # warmdown
        if self.warmdown > 0:
            warmdown = self.total_steps - int(self.total_steps * self.warmdown)
            if self._current_step >= warmdown:
                lr = -(self._current_step + 1 - warmdown) / (self.total_steps * self.warmdown)  * (self.base_lr - self.final_lr) + self.base_lr
                lr = np.min([lr, current_lr]) # if reduced by other scheduler
        
        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = np.max([lr, self.min_lr])
                
        self._current_step += 1