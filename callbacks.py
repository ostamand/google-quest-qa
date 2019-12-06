import tensorflow as tf
import tensorflow.keras.backend as K

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