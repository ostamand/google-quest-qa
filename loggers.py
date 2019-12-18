import os
from datetime import datetime

import wandb
from torch.utils.tensorboard import SummaryWriter

class MultiLogger():
    def __init__(self, model=None, log_dir='.logs', project=None, do_wandb=False, do_tb=False, **kwargs):
        self.do_wandb, self.do_tb = do_wandb, do_tb
        self.log_dir = log_dir
        self._writer = None

    def update(self, configs):
        if self.do_wandb:
            wandb.config.update(configs, allow_val_change=True)

    def add_scalars(self, logs):
        step = logs['step']
        logs = {k:v for k,v in logs.items() if k != 'step'}
        if self.do_tb:
            for k,v in logs.items():
                self.writer.add_scalar(k, v, step)
        if self.do_wandb:
            wandb.log(logs) # , step=step

    def close(self):
        if self.writer:
            self.writer.close()

    @property
    def writer(self):
        if self.do_tb and not self._writer:
            self._writer = SummaryWriter(os.path.join(self.log_dir, datetime.now().strftime("%Y%m%d_%H%M%S")))
        return self._writer
