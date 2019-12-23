import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from scipy.stats import spearmanr
import wandb

# taken from: https://www.kaggle.com/akensert/bert-base-tf2-0-minimalistic
def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)

class SpearmanrCallback(tf.keras.callbacks.Callback):
    """
    TODO: decrease lr when val rho gets worst
    """
    def __init__(self, val_data, patience_lr=5, patience_early=10, lr_scale=0.1, lr_min=1e-6, restore=True, do_wandb=False):
        self.x_valid, self.y_valid = val_data
        self.patience = {'lr': patience_lr, 'early': patience_early}
        self.restore, self.do_wandb = restore, do_wandb
        self.lr_scale, self.lr_min = lr_scale, lr_min

        self._reset()
        
    def _reset(self):
        self.best_rho = -np.inf
        self.worst = {'lr': 0, 'early': 0}
        
    def on_train_begin(self, logs={}):
        self._reset()
        
    def on_epoch_end(self, epoch, logs={}):
        y_preds = self.model.predict(self.x_valid, batch_size=8)
        rho_val = compute_spearmanr(self.y_valid, y_preds)

        bce = tf.keras.losses.BinaryCrossentropy()
        loss_val = bce(self.y_valid, y_preds).numpy()

        print(f"loss: {loss_val:.4f} (val), rho: {rho_val:.4f} (val)")

        if self.do_wandb:
             wandb.log({'loss/valid': loss_val, 'spearmanr/valid': rho_val})

        if rho_val > self.best_rho:
            self.best_rho = rho_val
            self.model.save_weights('.tmp/best_weights.h5')
            self.worst = {k: 0 for k, v in self.worst.items()}
        else:
            self.worst = {k: v+1 for k,v in self.worst.items()}

        # reduce on plateau
        if self.worst['lr'] >= self.patience['lr']:
            lr = K.get_value(self.model.optimizer.lr)
            print(f"\nReducing lr to {lr}")
            K.set_value(self.model.optimizer.lr, max(lr * self.lr_scale, self.lr_min))
            self.worst['lr'] = 0 

        # early stopping 
        if self.worst['early'] >= self.patience['early']:
            self.model.stop_training = True
            print(f'\nEarly stopping at epoch {epoch}')
            if self.restore:
                print(f'\nRestoring best weights')
                self.model.load_weights('.tmp/best_weights.h5')
            
        print(f"\nrho val: {rho_val:.4f}")

class LROneCycle(tf.keras.callbacks.Callback):
    """One Cycle Learning Rate Schedule Callback
    """
    def __init__(self, total_steps, min_lr=1e-5, up=0.5, down=0.5, do_wandb=False):
        super(LROneCycle, self).__init__()
        self.up, self.down = up, down
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.do_wandb = do_wandb
        
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
        if self.do_wandb:
            wandb.log({'train/lr': lr}, step=self.step)

        K.set_value(self.model.optimizer.lr, lr)
        
        self.step += 1
    
    def reset(self):
        self.prev_lr = None
        self.step = 0
        self.lrs = []