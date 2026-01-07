import torch
import torch.optim as optim
from typing import Dict, Any, List, Optional
import math


def create_optimizer(params, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    创建优化器
    
    Args:
        params: 模型参数
        config: 优化器配置字典
        
    Returns:
        optimizer: 优化器实例
    """
    optimizer_type = config.get('type', 'adamw')
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 1e-5)
    
    if optimizer_type == 'adam':
        betas = config.get('betas', (0.9, 0.999))
        eps = config.get('eps', 1e-8)
        optimizer = optim.Adam(
            params, 
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'adamw':
        betas = config.get('betas', (0.9, 0.999))
        eps = config.get('eps', 1e-8)
        optimizer = optim.AdamW(
            params, 
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'sgd':
        momentum = config.get('momentum', 0.9)
        dampening = config.get('dampening', 0)
        nesterov = config.get('nesterov', False)
        optimizer = optim.SGD(
            params, 
            lr=lr, 
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'rmsprop':
        alpha = config.get('alpha', 0.99)
        eps = config.get('eps', 1e-8)
        momentum = config.get('momentum', 0)
        centered = config.get('centered', False)
        optimizer = optim.RMSprop(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            momentum=momentum,
            centered=centered,
            weight_decay=weight_decay
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


class LookaheadOptimizer:
    """Lookahead优化器包装器"""
    
    def __init__(self, 
                 base_optimizer: torch.optim.Optimizer,
                 k: int = 5,
                 alpha: float = 0.5):
        
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.counter = 0
        
        # 保存fast weights的副本
        self.slow_weights = []
        for group in base_optimizer.param_groups:
            for p in group['params']:
                self.slow_weights.append(p.data.clone())
    
    def step(self, closure=None):
        """执行一步优化"""
        loss = self.base_optimizer.step(closure)
        self.counter += 1
        
        # 每k步更新一次slow weights
        if self.counter % self.k == 0:
            self._update_slow_weights()
        
        return loss
    
    def _update_slow_weights(self):
        """更新slow weights"""
        idx = 0
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # slow_weights = slow_weights + alpha * (fast_weights - slow_weights)
                slow_weight = self.slow_weights[idx]
                slow_weight.add_(self.alpha * (p.data - slow_weight))
                p.data.copy_(slow_weight)
                
                idx += 1
    
    def zero_grad(self):
        """清空梯度"""
        self.base_optimizer.zero_grad()
    
    def state_dict(self):
        """获取状态字典"""
        state = {
            'base_optimizer': self.base_optimizer.state_dict(),
            'counter': self.counter,
            'slow_weights': self.slow_weights
        }
        return state
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.counter = state_dict['counter']
        self.slow_weights = state_dict['slow_weights']


class RAdam(optim.Optimizer):
    """Rectified Adam优化器"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdam, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 衰减项
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # 计算自适应学习率
                if state['step'] > 4:
                    variance = exp_avg_sq / bias_correction2
                    variance_sqrt = variance.sqrt().add_(group['eps'])
                    
                    rho_inf = 2 / (1 - beta2) - 1
                    rho_t = rho_inf - 2 * state['step'] * beta2 ** state['step'] / (1 - beta2 ** state['step'])
                    
                    if rho_t > 4:
                        r_t = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                        adaptive_lr = math.sqrt(bias_correction1) * r_t / variance_sqrt
                        
                        p.data.addcdiv_(exp_avg, variance_sqrt, value=-group['lr'] * adaptive_lr)
                    else:
                        p.data.add_(exp_avg / bias_correction1, alpha=-group['lr'])
                else:
                    p.data.add_(exp_avg / bias_correction1, alpha=-group['lr'])
                
                # 权重衰减
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
        
        return loss