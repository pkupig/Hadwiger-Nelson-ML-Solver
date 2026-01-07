import torch
import torch.optim as optim
import math
import numpy as np
from typing import Dict, Any, List, Optional


def create_scheduler(optimizer, config: Dict[str, Any]) -> optim.lr_scheduler._LRScheduler:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 调度器配置字典
        
    Returns:
        scheduler: 学习率调度器实例
    """
    scheduler_type = config.get('type', 'cosine_annealing_warm_restarts')
    
    if scheduler_type == 'cosine_annealing_warm_restarts':
        T_0 = config.get('T_0', 100)
        T_mult = config.get('T_mult', 2)
        eta_min = config.get('eta_min', 1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=T_0, 
            T_mult=T_mult, 
            eta_min=eta_min
        )
    
    elif scheduler_type == 'cosine_annealing':
        T_max = config.get('T_max', 100)
        eta_min = config.get('eta_min', 0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=T_max, 
            eta_min=eta_min
        )
    
    elif scheduler_type == 'step':
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=gamma
        )
    
    elif scheduler_type == 'multi_step':
        milestones = config.get('milestones', [100, 200, 300])
        gamma = config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=milestones, 
            gamma=gamma
        )
    
    elif scheduler_type == 'exponential':
        gamma = config.get('gamma', 0.95)
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=gamma
        )
    
    elif scheduler_type == 'reduce_on_plateau':
        mode = config.get('mode', 'min')
        factor = config.get('factor', 0.1)
        patience = config.get('patience', 10)
        threshold = config.get('threshold', 1e-4)
        threshold_mode = config.get('threshold_mode', 'rel')
        cooldown = config.get('cooldown', 0)
        min_lr = config.get('min_lr', 0)
        eps = config.get('eps', 1e-8)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps
        )
    
    elif scheduler_type == 'cyclic':
        base_lr = config.get('base_lr', 1e-4)
        max_lr = config.get('max_lr', 1e-2)
        step_size_up = config.get('step_size_up', 2000)
        step_size_down = config.get('step_size_down', 2000)
        mode = config.get('mode', 'triangular')
        gamma = config.get('gamma', 1.0)
        scale_fn = config.get('scale_fn', None)
        scale_mode = config.get('scale_mode', 'cycle')
        cycle_momentum = config.get('cycle_momentum', True)
        
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            mode=mode,
            gamma=gamma,
            scale_fn=scale_fn,
            scale_mode=scale_mode,
            cycle_momentum=cycle_momentum
        )
    
    elif scheduler_type == 'one_cycle':
        max_lr = config.get('max_lr', 1e-2)
        total_steps = config.get('total_steps', None)
        epochs = config.get('epochs', None)
        steps_per_epoch = config.get('steps_per_epoch', None)
        pct_start = config.get('pct_start', 0.3)
        anneal_strategy = config.get('anneal_strategy', 'cos')
        cycle_momentum = config.get('cycle_momentum', True)
        base_momentum = config.get('base_momentum', 0.85)
        max_momentum = config.get('max_momentum', 0.95)
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


class CosineAnnealingWithWarmup(optim.lr_scheduler._LRScheduler):
    """带热身的余弦退火调度器"""
    
    def __init__(self, 
                 optimizer, 
                 T_max, 
                 warmup_steps=0,
                 warmup_lr_init=1e-5,
                 eta_min=0, 
                 last_epoch=-1):
        
        self.T_max = T_max
        self.warmup_steps = warmup_steps
        self.warmup_lr_init = warmup_lr_init
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 热身阶段
            lr_scale = (self.last_epoch + 1) / self.warmup_steps
            return [self.warmup_lr_init + (base_lr - self.warmup_lr_init) * lr_scale 
                   for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_steps) / (self.T_max - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_decay 
                   for base_lr in self.base_lrs]


class LinearWarmupCosineAnnealing(optim.lr_scheduler._LRScheduler):
    """线性热身 + 余弦退火"""
    
    def __init__(self, 
                 optimizer, 
                 warmup_epochs, 
                 max_epochs, 
                 warmup_start_lr=0.0,
                 eta_min=0.0, 
                 last_epoch=-1):
        
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性热身
            lr_scale = (self.last_epoch + 1) / self.warmup_epochs
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * lr_scale 
                   for base_lr in self.base_lrs]
        else:
            # 余弦退火
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_decay 
                   for base_lr in self.base_lrs]