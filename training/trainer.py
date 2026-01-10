import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import os
import time
from dataclasses import dataclass, field
import math
import random


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 5000
    batch_size: int = 4096
    num_train_pairs: int = 1000000
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    validation_freq: int = 100
    save_freq: int = 500
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    device: str = "cuda"
    seed: int = 42
    
    # 模拟退火配置
    use_simulated_annealing: bool = True
    sa_config: dict = field(default_factory=lambda: {
        'sa_freq': 10,
        'sa_steps': 3,
        'sa_T_init': 0.5,
        'sa_T_min': 1e-4,
        'sa_alpha': 0.95,
        'sa_perturb_std': 0.005
    })
    anneal_epochs: int = 1000
    min_entropy_weight: float = 0.01
    max_entropy_weight: float = 0.1


class _SAOptimizerWrapper:
    """
    模拟退火优化器包装器
    """
    
    def __init__(self, base_optimizer, model, loss_fn, sa_config):
        self.base_optimizer = base_optimizer
        self.model = model
        self.loss_fn = loss_fn
        self.sa_config = sa_config
        
        # SA参数
        self.sa_freq = sa_config.get('sa_freq', 10)
        self.sa_steps = sa_config.get('sa_steps', 3)
        self.sa_T_init = sa_config.get('sa_T_init', 0.5)
        self.sa_T_min = sa_config.get('sa_T_min', 1e-4)
        self.sa_alpha = sa_config.get('sa_alpha', 0.95)
        self.sa_perturb_std = sa_config.get('sa_perturb_std', 0.005)
        
        # 状态
        self.T_curr = self.sa_T_init
        self.step_count = 0
        self.best_state = None
        self.best_loss = float('inf')
        
        # 初始最佳状态 - 确保使用float64
        self.best_state = {k: v.clone().to(dtype=torch.float64) for k, v in model.state_dict().items()}
        
    def _simulated_annealing_step(self, p1, p2):
        """执行一步模拟退火"""
        device = p1.device
        
        # 保存当前状态 - 确保使用float64
        old_state = {k: v.clone().to(dtype=torch.float64) for k, v in self.model.state_dict().items()}
        
        # 计算当前损失
        with torch.no_grad():
            out1 = self.model(p1)
            out2 = self.model(p2)
            current_loss, _ = self.loss_fn(out1, out2)
            current_loss_val = current_loss.item()
        
        # 随机扰动参数 - 确保噪声是float64且在正确设备上
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    # 严格匹配参数的dtype和device
                    noise = torch.randn_like(param) * self.sa_perturb_std
                    param.data.add_(noise)
        
        # 计算扰动后损失
        with torch.no_grad():
            out1 = self.model(p1)
            out2 = self.model(p2)
            new_loss, _ = self.loss_fn(out1, out2)
            new_loss_val = new_loss.item()
        
        # Metropolis准则
        delta = new_loss_val - current_loss_val
        
        if delta < 0 or random.random() < math.exp(-delta / self.T_curr):
            # 接受新状态
            current_loss_val = new_loss_val
            
            # 更新最佳状态
            if new_loss_val < self.best_loss:
                self.best_loss = new_loss_val
                self.best_state = {k: v.clone().to(dtype=torch.float64) for k, v in self.model.state_dict().items()}
        else:
            # 拒绝新状态，恢复旧状态
            self.model.load_state_dict(old_state)
        
        # 降低温度
        self.T_curr = max(self.sa_T_min, self.T_curr * self.sa_alpha)
        
        return current_loss_val
    
    def step(self, p1=None, p2=None, closure=None):
        """执行一步优化"""
        # 首先执行基础优化器步骤
        if hasattr(self.base_optimizer, 'step'):
            if closure is not None:
                self.base_optimizer.step(closure)
            else:
                self.base_optimizer.step()
        
        # 定期执行模拟退火
        self.step_count += 1
        
        if p1 is not None and p2 is not None and self.step_count % self.sa_freq == 0:
            # 确保输入数据是float64
            if p1.dtype != torch.float64:
                p1 = p1.to(dtype=torch.float64)
            if p2.dtype != torch.float64:
                p2 = p2.to(dtype=torch.float64)
            
            for _ in range(self.sa_steps):
                self._simulated_annealing_step(p1, p2)
    
    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)
    
    def state_dict(self):
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'step_count': self.step_count,
            'T_curr': self.T_curr,
            'best_loss': self.best_loss,
            'sa_config': self.sa_config
        }
    
    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.step_count = state_dict.get('step_count', 0)
        self.T_curr = state_dict.get('T_curr', self.sa_T_init)
        self.best_loss = state_dict.get('best_loss', float('inf'))
        self.sa_config = state_dict.get('sa_config', {})
    
    def __getattr__(self, name):
        # 重定向到基础优化器的属性
        return getattr(self.base_optimizer, name)


class HadwigerNelsonTrainer:
    """Hadwiger-Nelson问题训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 data_generator,
                 loss_fn,
                 config: TrainingConfig):
        
        self.model = model
        self.data_generator = data_generator
        self.loss_fn = loss_fn
        self.config = config
        
        # 设置设备
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 1. 确保模型是float64并移动到设备
        # 这里的顺序很重要：先转dtype，再转device
        self.model = self.model.to(dtype=torch.float64).to(self.device)
        
        # 2. 创建优化器（必须在模型移动之后）
        self.optimizer = self._create_optimizer()
        
        # 3. 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self._get_base_optimizer(),
            T_0=100,
            T_mult=2,
            eta_min=1e-6
        )
        
        # 日志
        os.makedirs(config.log_dir, exist_ok=True)
        self.writer = SummaryWriter(config.log_dir)
        
        # 检查点目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_violation_rate = float('inf')
        self.train_history = []
        
        # 设置随机种子
        if config.seed is not None:
            self._set_seed(config.seed)
    
    def _get_base_optimizer(self):
        """获取基础优化器（如果是包装器，则返回基础优化器）"""
        if hasattr(self.optimizer, 'base_optimizer'):
            return self.optimizer.base_optimizer
        return self.optimizer
    
    def _create_optimizer(self):
        """创建优化器"""
        # 关键修复：设置 foreach=False 以避免 PyTorch 在 float64 下的 bug
        # eps=1e-8 保持默认，增加数值稳定性
        
        base_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-8,
            foreach=False  # <--- 这里的修改解决了 RuntimeError
        )
        
        if not self.config.use_simulated_annealing:
            return base_optimizer
        
        print(f"[Trainer] Enabling Simulated Annealing")
        return _SAOptimizerWrapper(
            base_optimizer=base_optimizer,
            model=self.model,
            loss_fn=self.loss_fn,
            sa_config=self.config.sa_config
        )
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = []
        epoch_conflict_losses = []
        epoch_entropy_losses = []
        
        num_batches = max(1, 100000 // self.config.batch_size)
        
        for batch_idx in range(num_batches):
            # 获取数据
            p1, p2 = self.data_generator.get_batch()
            # 确保数据是float64并移动到设备
            p1 = p1.to(self.device, dtype=torch.float64)
            p2 = p2.to(self.device, dtype=torch.float64)
            
            # 前向传播
            out1 = self.model(p1)
            out2 = self.model(p2)
            
            # 计算损失
            total_loss, loss_dict = self.loss_fn(out1, out2)
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
            
            # 优化步骤
            if self.config.use_simulated_annealing:
                self.optimizer.step(p1, p2)
            else:
                self.optimizer.step()

            # 熵退火
            progress = min(1.0, self.current_epoch / self.config.anneal_epochs)
            entropy_weight = self.config.min_entropy_weight + 0.5 * (self.config.max_entropy_weight - self.config.min_entropy_weight) * (1 - math.cos(math.pi * progress))
            if hasattr(self.loss_fn, 'set_entropy_weight'):
                self.loss_fn.set_entropy_weight(entropy_weight)
            
            # 记录损失
            epoch_losses.append(total_loss.item())
            epoch_conflict_losses.append(loss_dict.get('conflict', 0).item())
            epoch_entropy_losses.append(loss_dict.get('entropy', 0).item())
        
        # 更新学习率
        self.scheduler.step()
        
        # 计算平均损失
        avg_loss = np.mean(epoch_losses)
        avg_conflict = np.mean(epoch_conflict_losses)
        avg_entropy = np.mean(epoch_entropy_losses)
        current_lr = self._get_base_optimizer().param_groups[0]['lr']
        
        return {
            'total_loss': avg_loss,
            'conflict_loss': avg_conflict,
            'entropy_loss': avg_entropy,
            'learning_rate': current_lr
        }
    
    def validate(self, num_samples: int = 10000) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        with torch.no_grad():
            total_violations = 0
            total_pairs = 0
            
            for _ in range(num_samples // self.config.batch_size + 1):
                p1, p2 = self.data_generator.get_batch()
                p1 = p1.to(self.device, dtype=torch.float64)
                p2 = p2.to(self.device, dtype=torch.float64)
                
                colors1 = self.model.get_color_assignment(p1)
                colors2 = self.model.get_color_assignment(p2)
                
                violations = (colors1 == colors2).sum().item()
                total_violations += violations
                total_pairs += len(p1)
            
            violation_rate = total_violations / total_pairs if total_pairs > 0 else 0
            
            # 验证损失
            p1, p2 = self.data_generator.get_batch()
            p1 = p1.to(self.device, dtype=torch.float64)
            p2 = p2.to(self.device, dtype=torch.float64)
            out1 = self.model(p1)
            out2 = self.model(p2)
            val_loss, _ = self.loss_fn(out1, out2)
        
        return {
            'violation_rate': violation_rate * 100,
            'validation_loss': val_loss.item()
        }
    
    def train(self, num_epochs: Optional[int] = None):
        """主训练循环"""
        if num_epochs is None:
            num_epochs = self.config.epochs
        
        print(f"开始训练 {num_epochs} 个epochs...")
        print(f"设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"模型数据类型: {next(self.model.parameters()).dtype}")
        print(f"优化器类型: {type(self.optimizer)}")
        print(f"AdamW foreach模式: False") # 确认修复生效
        print("-" * 80)
        
        start_time = time.time()
        start_epoch = self.current_epoch
        target_end_epoch = start_epoch + num_epochs
        
        for epoch in range(start_epoch, target_end_epoch):
            self.current_epoch = epoch + 1
            train_metrics = self.train_epoch()
            
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            if (epoch + 1) % self.config.validation_freq == 0 or epoch == 0:
                val_metrics = self.validate()
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'Validation/{key}', value, epoch)
                
                print(f"Epoch {epoch+1:04d}/{target_end_epoch:04d} | "
                      f"Train Loss: {train_metrics['total_loss']:.6f} | "
                      f"Conflict: {train_metrics['conflict_loss']:.6f} | "
                      f"Val Violation: {val_metrics['violation_rate']:.2f}% | "
                      f"LR: {train_metrics['learning_rate']:.6f}")
                
                if val_metrics['violation_rate'] < self.best_loss:
                    self.best_loss = val_metrics['violation_rate']
                    self.save_checkpoint(f"best_model_epoch{epoch+1}.pth")
            
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch+1}.pth")
            
            self.train_history.append({
                'epoch': epoch + 1,
                'train': train_metrics,
                'validation': val_metrics if (epoch + 1) % self.config.validation_freq == 0 else None
            })
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总时间: {total_time:.2f}秒")
        print(f"最佳验证冲突率: {self.best_loss:.2f}%")
        self.save_checkpoint("final_model.pth")
        return self.train_history
    
    def save_checkpoint(self, filename: str):
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_history': self.train_history,
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.train_history = checkpoint.get('train_history', [])
        print(f"从 epoch {self.current_epoch} 加载检查点")

class MultiDimensionTrainer:
    """多维度训练管理器"""
    def __init__(self, dims: List[int], colors_per_dim: Dict[int, List[int]], base_config: TrainingConfig):
        self.dims = dims
        self.colors_per_dim = colors_per_dim
        self.base_config = base_config
        self.results = {} 
    
    def plot_comparison(self, save_path: str = "dimension_comparison.png") -> plt.Figure:
        if not self.results: return None
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ax1 = axes[0]
        colors = ['b', 'g', 'r', 'c', 'm']
        markers = ['o', 's', '^', 'D', 'v']
        
        for idx, dim in enumerate(sorted(self.results.keys())):
            dim_results = self.results[dim]
            k_values = sorted(dim_results.keys())
            violation_rates = [dim_results[k]['final_violation_rate'] for k in k_values]
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax1.plot(k_values, violation_rates, f'{color}{marker}-', linewidth=2, markersize=8, label=f'{dim}D')
            
        ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='1% Threshold')
        ax1.set_xlabel('Number of colors (k)')
        ax1.set_ylabel('Violation Rate (%)')
        ax1.set_title('Violation Rate vs. Colors across Dimensions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        dims_list = []
        min_feasible_k = []
        for dim in sorted(self.results.keys()):
            dim_results = self.results[dim]
            feasible = [k for k, res in dim_results.items() if res['final_violation_rate'] < 1.0]
            if feasible:
                dims_list.append(dim)
                min_feasible_k.append(min(feasible))
            else:
                dims_list.append(dim)
                min_feasible_k.append(max(dim_results.keys()) + 1)
        
        if dims_list:
            bars = ax2.bar(dims_list, min_feasible_k, color='purple', alpha=0.6)
            ax2.set_xlabel('Dimension')
            ax2.set_ylabel('Estimated Chromatic Number (Upper Bound)')
            ax2.set_title('Chromatic Number Bounds by Dimension')
            ax2.set_xticks(dims_list)
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        if hasattr(self.base_config, 'log_dir'):
            full_save_path = os.path.join(self.base_config.log_dir, save_path)
        else:
            full_save_path = save_path
        plt.savefig(full_save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {full_save_path}")
        return fig