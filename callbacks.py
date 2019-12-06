import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from scipy.stats import spearmanr

# taken from: https://www.kaggle.com/akensert/bert-base-tf2-0-minimalistic
def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)

class SpearmanrCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, patience=5, restore=True):
        self.x_valid, self.y_valid = val_data
        self.patience = patience
        self.restore = restore
        
        self._reset()
        
    def _reset(self):
        self.best_rho = -np.inf
        self.worst = 0
        
    def on_train_begin(self, logs={}):
        self._reset()
        
    def on_epoch_end(self, epoch, logs={}):
        y_preds = self.model.predict(self.x_valid, batch_size=8)
        rho_val = compute_spearmanr(self.y_valid, y_preds)
        
        if rho_val > self.best_rho:
            self.best_rho = rho_val
            self.model.save_weights('best_weights.h5')
            self.worst = 0 
        else:
            self.worst += 1
        
        if self.worst >= self.patience:
            self.model.stop_training = True
            print(f'\nEarly stopping at epoch {epoch}')
            if self.restore:
                print(f'\nRestoring best weights')
                self.model.load_weights('best_weights.h5')
            
        print(f"\nrho val: {rho_val:.4f}")

class LROneCycle(tf.keras.callbacks.Callback):
    """One Cycle Learning Rate Schedule Callback
    """
    def __init__(self, total_steps, min_lr=1e-5, up=0.5, down=0.5):
        super(LROneCycle, self).__init__()
        self.up, self.down = up, down
        self.total_steps = total_steps
        self.min_lr = min_lr
        
        self.step_up = int(self.up * self.total_steps)
        self.step_down = int(self.down * self.total_steps) 
        
        self._lr = None
        
    @property 
    def lr(self):
        if self._lr is None:
            self._lr = K.get_value(self.model.optimizer.lr)
        return self._lr
        
    def on_train_begin(self, logs=None):
        self.reset()
    
    def on_batch_begin(self, batch, logs):
        lr = self.lr
        if self.step < self.step_up:
            lr = self.min_lr + (self.lr - self.min_lr) * (self.step / self.step_up)
        if self.step > self.total_steps - self.step_down:
            lr = self.lr - (self.lr - self.min_lr) * ((self.step-(self.total_steps-self.step_down))/(self.step_down-1))
        lr = lr.astype(np.float32)

        # in case reduced by other callback
        current_lr = K.get_value(self.model.optimizer.lr)
        if self.prev_lr and current_lr < self.prev_lr:
            lr = min(current_lr, lr)
            self.prev_lr = lr if lr < current_lr else self.prev_lr
        else:
            self.prev_lr = lr
      
        # log lr per step
        self.lrs.append(lr)
        K.set_value(self.model.optimizer.lr, lr)
        
        self.step += 1
    
    def reset(self):
        self.prev_lr = None
        self.step = 0
        self.lrs = []