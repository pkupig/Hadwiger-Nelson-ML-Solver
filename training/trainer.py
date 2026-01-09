import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
import time
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 5000
    batch_size: int = 4096
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    validation_freq: int = 100
    save_freq: int = 500
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    device: str = "cuda"
    seed: int = 42


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
        self.model = self.model.to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
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
        self.train_history = []
        
        # 设置随机种子
        if config.seed is not None:
            self._set_seed(config.seed)
    
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
        
        # 假设每个epoch处理固定数量的点对
        num_batches = max(1, 100000 // self.config.batch_size)
        
        for batch_idx in range(num_batches):
            # 获取数据
            p1, p2 = self.data_generator.get_batch()
            p1 = p1.to(self.device)
            p2 = p2.to(self.device)
            
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
            
            self.optimizer.step()
            
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
        
        return {
            'total_loss': avg_loss,
            'conflict_loss': avg_conflict,
            'entropy_loss': avg_entropy,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, num_samples: int = 10000) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        with torch.no_grad():
            total_violations = 0
            total_pairs = 0
            
            for _ in range(num_samples // self.config.batch_size + 1):
                # 生成验证数据
                p1, p2 = self.data_generator.get_batch()
                p1 = p1.to(self.device)
                p2 = p2.to(self.device)
                
                # 获取颜色分配
                colors1 = self.model.get_color_assignment(p1)
                colors2 = self.model.get_color_assignment(p2)
                
                # 统计冲突
                violations = (colors1 == colors2).sum().item()
                total_violations += violations
                total_pairs += len(p1)
            
            violation_rate = total_violations / total_pairs if total_pairs > 0 else 0
            
            # 计算验证损失
            p1, p2 = self.data_generator.get_batch()
            p1 = p1.to(self.device)
            p2 = p2.to(self.device)
            
            out1 = self.model(p1)
            out2 = self.model(p2)
            val_loss, _ = self.loss_fn(out1, out2)
        
        return {
            'violation_rate': violation_rate * 100,  # 百分比
            'validation_loss': val_loss.item()
        }
    
    def train(self, num_epochs: Optional[int] = None):
        """主训练循环"""
        if num_epochs is None:
            num_epochs = self.config.epochs
        
        print(f"开始训练 {num_epochs} 个epochs...")
        print(f"设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 80)
        
        start_time = time.time()
        
        # 修正1：在循环外计算目标结束Epoch
        start_epoch = self.current_epoch
        target_end_epoch = start_epoch + num_epochs
        
        # 修正2：循环范围使用固定的 start 和 target
        for epoch in range(start_epoch, target_end_epoch):
            self.current_epoch = epoch + 1
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 记录到TensorBoard
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            # 验证
            if (epoch + 1) % self.config.validation_freq == 0 or epoch == 0:
                val_metrics = self.validate()
                
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'Validation/{key}', value, epoch)
                
                # 修正3：打印时使用固定的 target_end_epoch
                print(f"Epoch {epoch+1:04d}/{target_end_epoch:04d} | "
                      f"Train Loss: {train_metrics['total_loss']:.6f} | "
                      f"Conflict: {train_metrics['conflict_loss']:.6f} | "
                      f"Val Violation: {val_metrics['violation_rate']:.2f}% | "
                      f"LR: {train_metrics['learning_rate']:.6f}")
                
                # 保存最佳模型
                if val_metrics['violation_rate'] < self.best_loss:
                    self.best_loss = val_metrics['violation_rate']
                    self.save_checkpoint(f"best_model_epoch{epoch+1}.pth")
            
            # 定期保存检查点
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch+1}.pth")
            
            # 保存训练历史
            self.train_history.append({
                'epoch': epoch + 1,
                'train': train_metrics,
                'validation': val_metrics if (epoch + 1) % self.config.validation_freq == 0 else None
            })
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n训练完成！总时间: {total_time:.2f}秒")
        print(f"最佳验证冲突率: {self.best_loss:.2f}%")
        
        # 保存最终模型
        self.save_checkpoint("final_model.pth")
        
        return self.train_history
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
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
        print(f"检查点保存到: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
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
        print(f"最佳损失: {self.best_loss:.4f}")


class MultiDimensionTrainer:
    """多维度训练器：同时在2D、3D、4D上训练"""
    
    def __init__(self, 
                 dims: List[int] = [2, 3, 4],
                 colors_per_dim: Dict[int, List[int]] = None,
                 base_config: TrainingConfig = None):
        
        self.dims = dims
        
        if colors_per_dim is None:
            colors_per_dim = {
                2: [3, 4, 5, 6, 7, 8],
                3: [4, 5, 6, 7, 8, 9, 10],
                4: [5, 6, 7, 8, 9, 10, 11, 12]
            }
        self.colors_per_dim = colors_per_dim
        
        self.base_config = base_config or TrainingConfig()
        
        # 存储每个维度的训练器
        self.trainers = {}
        self.results = {}
    
    def run_all_experiments(self):
        """运行所有维度的实验"""
        print("=" * 80)
        print("开始多维度Hadwiger-Nelson实验")
        print("=" * 80)
        
        for dim in self.dims:
            print(f"\n{'='*60}")
            print(f"维度 {dim}D 实验")
            print('='*60)
            
            self.results[dim] = {}
            
            for k in self.colors_per_dim[dim]:
                print(f"\n测试颜色数 k = {k}")
                print('-' * 40)
                
                # 创建数据生成器
                from data.generator import PointPairGenerator
                generator = PointPairGenerator(
                    dim=dim,
                    batch_size=self.base_config.batch_size,
                    device=self.base_config.device
                )
                
                # 创建模型
                from models.mlp_mapper import MLPColorMapper
                model = MLPColorMapper(
                    input_dim=dim,
                    num_colors=k,
                    hidden_dims=[128, 256, 128]
                )
                
                # 创建损失函数
                from losses.constraint_loss import ConstraintLoss
                loss_fn = ConstraintLoss(
                    conflict_weight=1.0,
                    entropy_weight=0.1,
                    uniformity_weight=0.01
                )
                
                # 创建训练器
                trainer = HadwigerNelsonTrainer(
                    model=model,
                    data_generator=generator,
                    loss_fn=loss_fn,
                    config=self.base_config
                )
                
                # 训练
                history = trainer.train(num_epochs=min(self.base_config.epochs, 2000))
                
                # 最终验证
                final_metrics = trainer.validate(num_samples=50000)
                
                # 保存结果
                self.results[dim][k] = {
                    'final_violation_rate': final_metrics['violation_rate'],
                    'final_loss': final_metrics['validation_loss'],
                    'best_violation_rate': trainer.best_loss,
                    'history': history
                }
                
                print(f"最终冲突率: {final_metrics['violation_rate']:.2f}%")
        
        return self.results
    
    def plot_comparison(self):
        """绘制多维度比较图"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 各维度不同k值的最终冲突率
        ax1 = axes[0, 0]
        for dim in self.dims:
            k_values = sorted(self.results[dim].keys())
            violation_rates = [self.results[dim][k]['final_violation_rate'] for k in k_values]
            ax1.plot(k_values, violation_rates, 'o-', label=f'{dim}D', linewidth=2)
        
        ax1.set_xlabel('Number of colors (k)')
        ax1.set_ylabel('Violation Rate (%)')
        ax1.set_title('Violation Rate vs k for Different Dimensions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 固定k值，比较不同维度
        ax2 = axes[0, 1]
        common_k = [4, 5, 6, 7, 8]
        for k in common_k:
            dims = []
            rates = []
            for dim in self.dims:
                if k in self.results[dim]:
                    dims.append(dim)
                    rates.append(self.results[dim][k]['final_violation_rate'])
            if rates:
                ax2.plot(dims, rates, 's-', label=f'k={k}', linewidth=2)
        
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Violation Rate (%)')
        ax2.set_title('Violation Rate vs Dimension for Fixed k')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 训练曲线示例
        ax3 = axes[1, 0]
        if self.dims and self.colors_per_dim[self.dims[0]]:
            example_dim = self.dims[0]
            example_k = self.colors_per_dim[example_dim][-1]  # 取最大的k
            
            if example_k in self.results[example_dim]:
                history = self.results[example_dim][example_k]['history']
                epochs = [h['epoch'] for h in history]
                losses = [h['train']['total_loss'] for h in history]
                
                ax3.plot(epochs, losses, 'b-', linewidth=2)
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Training Loss')
                ax3.set_title(f'Training Curve: {example_dim}D, k={example_k}')
                ax3.grid(True, alpha=0.3)
        
        # 4. 颜色数下界估计
        ax4 = axes[1, 1]
        for dim in self.dims:
            k_values = sorted(self.results[dim].keys())
            feasible = []
            
            for k in k_values:
                violation_rate = self.results[dim][k]['final_violation_rate']
                # 如果冲突率小于阈值，认为可行
                if violation_rate < 1.0:  # 1%阈值
                    feasible.append(k)
            
            if feasible:
                estimated_lower_bound = min(feasible) if feasible else None
                ax4.bar(dim, estimated_lower_bound if estimated_lower_bound else 0, 
                       label=f'{dim}D est: {estimated_lower_bound}')
        
        ax4.set_xlabel('Dimension')
        ax4.set_ylabel('Estimated Lower Bound')
        ax4.set_title('Estimated Chromatic Number Lower Bound')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('multi_dimension_comparison.png', dpi=150, bbox_inches='tight')
        print("比较图已保存到 multi_dimension_comparison.png")
        
        return fig