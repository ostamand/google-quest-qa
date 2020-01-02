import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from apex import amp

from helpers_torch import set_seed
from helpers import compute_spearmanr, EarlyStoppingSimple
from schedulers import LearningRateWithUpDown
from loggers import MultiLogger

import pdb
    
def get_max_gradient(params, norm=2):
    with torch.no_grad():
        max_g = -np.inf
        for param in [param for param in params if param.requires_grad]:
            g = torch.norm(param.grad, norm).item()
            max_g = g if g > max_g else max_g
    return max_g

class Trainer():
    """
        - Mixed precision training
        - Accumulation steps
        - TensorBoard logging
        - Wandb logging
        - ReduceOnPlateau
        - Early stopping
        - LR Scheduling (warmup, warmdown)
    """
    def __init__(self, params=None, **kwargs):
        if not params:
            self.params = self.default_params()
        else:
            self.params = params
        self._setup_params(**kwargs)
        
    def _setup_params(self, **kwargs):
        if not kwargs:
            return 
        for k,v in kwargs.items():
            if k in self.params:
                self.params[k] = v
    
    def train(self, model, loaders, optimizer, params=None, **kwargs):
        rho_vals = []

        self._setup_params(**kwargs)
        p = self.params
        
        set_seed(p['seed'])
        
        train_loader = loaders['train']
        valid_loader = loaders['valid'] if 'valid' in loaders else None
        
        device = torch.device(p['device'])
        model.to(device)

        lossf = nn.BCEWithLogitsLoss()
        optimizer.zero_grad()
        running_loss = None
        acc_loss = 0
        it = 1 # global steps

        logger = MultiLogger(**p)
        logger.update(p) # will save all configs
        
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2, verbose=True, factor=0.1)
        early_stopping = EarlyStoppingSimple(model, patience=5, min_delta=0)  

        lr_scheduler = LearningRateWithUpDown(
            optimizer, 
            p['epochs'] * len(train_loader), 
            warmup=p['warmup'], 
            warmdown=p['warmdown'], 
            ini_lr=1e-6, 
            final_lr=1e-6
        )
    
        for epoch_i in range(p['epochs']):
            model.train()
        
            if early_stopping.training_done:
                print(f"Early stopping on epoch {epoch_i-1}")
                break
            
            pb = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch: {epoch_i+1}/{p['epochs']}")

            for batch_i, batch in pb:
                # step
                lr_scheduler.step()

                token_type_batch = None

                if len(batch) == 2:
                    x_batch, y_batch = batch
                if len(batch) == 3:
                    x_batch, token_type_batch, y_batch = batch
                    token_type_batch = token_type_batch.to(device)

                bs = x_batch.shape[0]

                outs = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), token_type_ids=token_type_batch)
                loss = lossf(outs, y_batch.to(device)) / p['accumulation_steps']

                acc_loss += loss.item() / bs

                if p['do_apex']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                    
                if it % p['accumulation_steps'] == 0:
                    logs = {'step': it}
                    logs['lr/train'] = optimizer.param_groups[0]['lr']

                    # clip gradients (l2)
                    if p['clip'] is not None:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), p['clip'])
                    
                    # log gradients after clipping (l2)
                    # logs['grads/train'] = get_max_gradient(model.parameters())

                    optimizer.step()
                    optimizer.zero_grad()
                    
                    logs['loss/train'] = acc_loss

                    logger.add_scalars(logs)
                    
                    if running_loss:
                        running_loss = 0.98 * running_loss + 0.02 * acc_loss
                    else:
                        running_loss = acc_loss

                    pb.set_postfix(loss = running_loss)
                    
                    acc_loss = 0
                
                it+=1
                
            # evaluate 
            if valid_loader is not None:
                metrics = self.evaluate(model, valid_loader)
                
                scheduler.step(metrics['spearmanr'])
                early_stopping.step(metrics['spearmanr'])

                logs = {'step': it}
                logs['loss/valid'] = metrics['loss']
                logs['spearmanr/valid'] = metrics['spearmanr']
                logger.add_scalars(logs)

                rho_vals.append(logs['spearmanr/valid'])
                
                print(f"rho: {metrics['spearmanr']:.4f} (val), loss: {metrics['loss']:.4f} (val)")
        
        if valid_loader:
            early_stopping.restore()

        logger.close()

        return rho_vals
    
    def evaluate(self, model, loader):
        device = torch.device(self.params['device'])
        model.eval()
        lossf = nn.BCEWithLogitsLoss()
        y_true, y_pred = [], []
        loss = 0
        for batch_i, batch in enumerate(loader):
            with torch.no_grad():

                token_type_batch = None
                if len(batch) == 2:
                    x_batch, y_batch = batch
                if len(batch) == 3:
                    x_batch, token_type_batch, y_batch = batch
                    token_type_batch = token_type_batch.to(device)

                outs = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), token_type_ids=token_type_batch)
                loss += lossf(outs, y_batch.to(device)).item()

                y_pred.append(torch.sigmoid(outs).cpu().numpy())
                y_true.append(y_batch.numpy())

        y_pred = np.vstack(y_pred)
        y_true = np.vstack(y_true)
        rho = compute_spearmanr(y_true, y_pred)
        logs = {'loss': loss / y_pred.shape[0], 'spearmanr': rho}
        model.train()
        return logs
        
    @classmethod
    def default_params(cls):
        return {
            'device': 'cuda', 
            'accumulation_steps': 2,
            'epochs': 1,
            'log_dir': '.logs',
            'seed': 42, 
            'clip': 1, 
            'warmup': 0.1, 
            'warmdown': 0.1,
            'do_apex': True,
            'do_wandb': False,
            'watch': False,
            'do_tb': False, 
            'project': None
        }