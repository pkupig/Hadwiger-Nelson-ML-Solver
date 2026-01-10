"""
难例挖掘器
主动寻找难以满足约束的点对，提高训练效率
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Dict
import heapq
import random
# 确保从 training.trainer 导入父类
from training.trainer import HadwigerNelsonTrainer

class HardExampleBank:
    """难例银行：存储和检索难以满足约束的点对"""
    
    def __init__(self, 
                 max_size: int = 50000,
                 min_loss_threshold: float = 0.1,
                 priority_weight: float = 0.5,  # 0=均匀采样，1=完全按损失采样
                 device: str = "cuda"):
        
        self.max_size = max_size
        self.min_loss_threshold = min_loss_threshold
        self.priority_weight = priority_weight
        self.device = device
        
        # 使用最大堆存储难例（损失大的在前）
        self.heap = []  # 存储(-loss, index)对
        self.examples = []  # 存储(p1, p2, loss)
        self.index = 0
        self.total_added = 0
        self.total_rejected = 0
    
    def add_batch(self, 
                  p1: torch.Tensor, 
                  p2: torch.Tensor, 
                  losses: torch.Tensor):
        """
        添加一批点到难例银行
        """
        # 转换为CPU numpy以存储
        p1_np = p1.detach().cpu().numpy()
        p2_np = p2.detach().cpu().numpy()
        losses_np = losses.detach().cpu().numpy()
        
        batch_size = len(p1_np)
        
        for i in range(batch_size):
            loss = float(losses_np[i])
            
            # 只有损失足够大的点对才被考虑
            if loss < self.min_loss_threshold:
                self.total_rejected += 1
                continue
            
            # 如果堆已满，替换最小的
            if len(self.heap) >= self.max_size:
                # 获取当前最小损失（因为是负值，所以是最大的负数）
                min_loss_in_heap = -self.heap[0][0]
                
                if loss > min_loss_in_heap:
                    # 弹出最小的
                    _, popped_idx = heapq.heappop(self.heap)
                    # 用新例子替换
                    self.examples[popped_idx] = (p1_np[i], p2_np[i], loss)
                    # 将新例子推入堆
                    heapq.heappush(self.heap, (-loss, popped_idx))
                    self.total_added += 1
            else:
                # 堆未满，直接添加
                self.examples.append((p1_np[i], p2_np[i], loss))
                heapq.heappush(self.heap, (-loss, len(self.examples) - 1))
                self.total_added += 1
    
    def sample_batch(self, 
                     batch_size: int,
                     strategy: str = "priority") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从难例银行中采样一批点对
        """
        if not self.examples:
            return None, None
        
        if strategy == "uniform":
            # 均匀采样
            indices = np.random.choice(
                len(self.examples), 
                size=min(batch_size, len(self.examples)),
                replace=False
            )
        
        elif strategy == "priority":
            # 按损失优先级采样
            losses = np.array([ex[2] for ex in self.examples])
            
            # 使用softmax创建概率分布
            # 温度参数控制分布的尖锐程度
            temperature = 1.0 / (self.priority_weight + 1e-8)
            weights = np.exp(losses / temperature)
            weights = weights / weights.sum()
            
            indices = np.random.choice(
                len(self.examples),
                size=min(batch_size, len(self.examples)),
                p=weights,
                replace=False
            )
        
        elif strategy == "mixed":
            # 混合策略：一部分难例，一部分随机
            num_hard = int(batch_size * 0.7)
            num_random = batch_size - num_hard
            
            # 难例部分
            hard_indices = self._sample_by_priority(num_hard)
            
            # 随机部分
            if num_random > 0:
                all_indices = set(range(len(self.examples)))
                remaining_indices = list(all_indices - set(hard_indices))
                if remaining_indices:
                    random_indices = np.random.choice(
                        remaining_indices,
                        size=min(num_random, len(remaining_indices)),
                        replace=False
                    )
                    indices = np.concatenate([hard_indices, random_indices])
                else:
                    indices = hard_indices
            else:
                indices = hard_indices
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # 收集点对
        p1_list = []
        p2_list = []
        
        for idx in indices:
            p1_np, p2_np, _ = self.examples[idx]
            p1_list.append(p1_np)
            p2_list.append(p2_np)
        
        # 转换为tensor - 强制使用torch.float64
        p1 = torch.tensor(np.stack(p1_list), dtype=torch.float64, device=self.device)
        p2 = torch.tensor(np.stack(p2_list), dtype=torch.float64, device=self.device)
        
        return p1, p2
    
    def _sample_by_priority(self, n: int) -> np.ndarray:
        """按优先级采样n个索引"""
        if not self.examples:
            return np.array([], dtype=int)
        
        losses = np.array([ex[2] for ex in self.examples])
        
        # 使用幂律分布
        # p(i) ∝ loss(i)^α
        alpha = 2.0  # 控制分布的尖锐程度
        weights = losses ** alpha
        weights = weights / weights.sum()
        
        return np.random.choice(
            len(self.examples),
            size=min(n, len(self.examples)),
            p=weights,
            replace=False
        )
    
    def get_hardest_examples(self, 
                            n: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取最难的n个例子"""
        if not self.examples:
            return None, None, None
        
        # 从堆中获取损失最大的n个
        sorted_items = sorted(self.heap, key=lambda x: -x[0])[:n]
        
        p1_list = []
        p2_list = []
        loss_list = []
        
        for neg_loss, idx in sorted_items:
            p1_np, p2_np, loss = self.examples[idx]
            p1_list.append(p1_np)
            p2_list.append(p2_np)
            loss_list.append(loss)
        
        # 强制使用torch.float64
        p1 = torch.tensor(np.stack(p1_list), dtype=torch.float64, device=self.device)
        p2 = torch.tensor(np.stack(p2_list), dtype=torch.float64, device=self.device)
        losses = torch.tensor(loss_list, dtype=torch.float64, device=self.device)
        
        return p1, p2, losses
    
    def analyze(self) -> Dict[str, float]:
        """分析难例银行的统计信息"""
        if not self.examples:
            return {
                "size": 0,
                "mean_loss": 0.0,
                "max_loss": 0.0,
                "min_loss": 0.0,
                "utilization": 0.0
            }
        
        losses = np.array([ex[2] for ex in self.examples])
        
        return {
            "size": len(self.examples),
            "mean_loss": float(losses.mean()),
            "max_loss": float(losses.max()),
            "min_loss": float(losses.min()),
            "std_loss": float(losses.std()),
            "total_added": self.total_added,
            "total_rejected": self.total_rejected,
            "utilization": len(self.examples) / self.max_size if self.max_size > 0 else 1.0
        }
    
    def clear(self):
        """清空难例银行"""
        self.heap.clear()
        self.examples.clear()
        self.index = 0
        self.total_added = 0
        self.total_rejected = 0


class AdaptiveHardExampleMiner:
    """自适应难例挖掘器"""
    
    def __init__(self, 
                 initial_threshold: float = 0.1,
                 update_frequency: int = 100,
                 decay_rate: float = 0.99,
                 min_threshold: float = 0.01,
                 device: str = "cuda"):
        
        self.threshold = initial_threshold
        self.initial_threshold = initial_threshold
        self.update_frequency = update_frequency
        self.decay_rate = decay_rate
        self.min_threshold = min_threshold
        self.device = device
        
        self.step_count = 0
        self.recent_losses = []
        self.max_recent_losses = 1000
    
    def should_keep_example(self, loss: float) -> bool:
        """判断是否应该保留这个例子"""
        self.step_count += 1
        
        # 记录最近的损失
        self.recent_losses.append(loss)
        if len(self.recent_losses) > self.max_recent_losses:
            self.recent_losses.pop(0)
        
        # 更新阈值
        if self.step_count % self.update_frequency == 0:
            self._update_threshold()
        
        return loss > self.threshold
    
    def _update_threshold(self):
        """更新阈值"""
        if not self.recent_losses:
            return
        
        # 计算最近损失的分位数
        recent_losses_np = np.array(self.recent_losses)
        percentile = 75  # 取75%分位数
        
        new_threshold = np.percentile(recent_losses_np, percentile)
        
        # 指数移动平均
        self.threshold = self.decay_rate * self.threshold + (1 - self.decay_rate) * new_threshold
        
        # 确保不低于最小阈值
        self.threshold = max(self.threshold, self.min_threshold)
    
    def get_threshold(self) -> float:
        """获取当前阈值"""
        return self.threshold


class GeometricHardExampleGenerator:
    """几何难例生成器：生成已知的硬配置"""
    
    def __init__(self, dim: int = 2, device: str = "cuda"):
        self.dim = dim
        self.device = device
    
    def generate_moser_spindle(self) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        生成Moser Spindle图（2D中的已知硬实例）
        """
        if self.dim != 2:
            return torch.empty(0, device=self.device), []
        
        # Moser spindle的7个点
        # 强制使用torch.float64
        points = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2],
            [1.5, np.sqrt(3)/2],
            [1.5, -np.sqrt(3)/2],
            [2.0, 0.0]
        ], dtype=torch.float64, device=self.device)
        
        # 距离为1的边
        edges = [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4),
            (3, 5), (4, 5), (4, 6), (5, 6), (2, 6), (3, 6)
        ]
        return points, edges
    
    def generate_golomb_graph(self) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        生成Golomb图
        """
        import cmath
        points_complex = []
        base_points = [0, 1, 1j]
        for point in base_points:
            for angle in [0, 90, 180, 270]:
                rad = np.deg2rad(angle)
                rotated = point * cmath.exp(1j * rad)
                points_complex.append(rotated)
        
        unique_points = []
        for p in points_complex:
            coord = (round(p.real, 6), round(p.imag, 6))
            if coord not in unique_points:
                unique_points.append(coord)
        
        # 强制使用torch.float64
        points = torch.tensor(unique_points, dtype=torch.float64, device=self.device)
        
        edges = []
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                dist = torch.norm(points[i] - points[j])
                if abs(dist - 1.0) < 1e-6:
                    edges.append((i, j))
        
        return points, edges


class HardExampleTrainer(HadwigerNelsonTrainer):
    """
    带难例挖掘的训练器 (继承自标准训练器)
    集成主动学习：在训练过程中自动识别并重点训练困难样本
    """
    
    def __init__(self, 
                 model,
                 data_generator,
                 loss_fn,
                 config, 
                 hard_example_bank=None,
                 mining_frequency: int = 10,
                 hard_example_ratio: float = 0.3,
                 device: str = "cuda"):
        
        # 1. 初始化父类 (复用基础功能)
        super().__init__(model, data_generator, loss_fn, config)
        
        # 2. 初始化难例挖掘特有的组件
        if hard_example_bank is None:
            self.hard_example_bank = HardExampleBank(device=self.device)
        else:
            self.hard_example_bank = hard_example_bank
            
        if hasattr(config, 'data') and isinstance(config.data, dict):
            he_config = config.data.get('hard_examples', {})
            self.mining_frequency = he_config.get('mining_iterations', mining_frequency)
        else:
            self.mining_frequency = mining_frequency
            
        self.hard_example_ratio = hard_example_ratio
        
        self.adaptive_miner = AdaptiveHardExampleMiner(device=self.device)
        self.geometric_generator = GeometricHardExampleGenerator(
            dim=data_generator.dim,
            device=self.device
        )
        
        self.step_count = 0

    def train_epoch(self) -> Dict[str, float]:
        """
        重写父类的 train_epoch 以注入难例挖掘逻辑
        """
        self.model.train()
        
        # 计算温度退火
        if hasattr(self.config, 'anneal_epochs'):
            import math
            progress = min(1.0, self.current_epoch / self.config.anneal_epochs)
            if hasattr(self.loss_fn, 'set_entropy_weight'):
                entropy_weight = self.config.min_entropy_weight + 0.5 * (self.config.max_entropy_weight - self.config.min_entropy_weight) * (1 - math.cos(math.pi * progress))
                self.loss_fn.set_entropy_weight(entropy_weight)
            
        epoch_loss = 0
        loss_stats = {}
        
        num_batches = max(1, 100000 // self.config.batch_size)
        
        for batch_idx in range(num_batches):
            self.step_count += 1
            
            # --- 难例挖掘数据混合逻辑 ---
            use_hard_examples = (self.step_count % self.mining_frequency == 0)
            batch_size = self.config.batch_size
            
            p1, p2 = None, None
            
            if use_hard_examples and self.hard_example_bank.analyze()["size"] > batch_size // 10:
                # 使用部分难例
                num_hard = int(batch_size * self.hard_example_ratio)
                num_random = batch_size - num_hard
                
                # 从难例银行采样 (银行会返回float64的tensor)
                p1_hard, p2_hard = self.hard_example_bank.sample_batch(
                    num_hard, strategy="priority"
                )
                
                # 生成随机样本
                p1_random, p2_random = self.data_generator.get_batch()
                # 截取需要的数量
                if p1_random.shape[0] > num_random:
                    p1_random = p1_random[:num_random]
                    p2_random = p2_random[:num_random]
                
                # 合并数据
                if p1_hard is not None:
                    # 确保随机样本的类型与难例一致 (float64)
                    p1_random = p1_random.to(dtype=torch.float64)
                    p2_random = p2_random.to(dtype=torch.float64)
                    
                    p1 = torch.cat([p1_hard, p1_random], dim=0)
                    p2 = torch.cat([p2_hard, p2_random], dim=0)
                else:
                    p1, p2 = p1_random, p2_random
            else:
                # 只使用随机样本
                p1, p2 = self.data_generator.get_batch()
            
            # 确保数据是float64并移动到设备
            p1 = p1.to(device=self.device, dtype=torch.float64)
            p2 = p2.to(device=self.device, dtype=torch.float64)
            
            # --- 前向传播 ---
            out1 = self.model(p1)
            out2 = self.model(p2)
            
            # 计算损失
            total_loss, batch_loss_dict = self.loss_fn(out1, out2)
            
            # --- 在线难例挖掘 (Online Mining) ---
            with torch.no_grad():
                # 计算逐点冲突程度 (dot product)
                conflict = torch.sum(out1 * out2, dim=1)
                
                # 判断是否为难例
                hard_mask = conflict > self.adaptive_miner.get_threshold()
                
                if hard_mask.any():
                    # 将发现的新难例存入银行
                    self.hard_example_bank.add_batch(
                        p1[hard_mask], 
                        p2[hard_mask], 
                        conflict[hard_mask]
                    )
            
            # --- 反向传播 ---
            self.optimizer.zero_grad()
            total_loss.backward()
            
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            # --- 优化步骤 ---
            # 如果使用SA，尝试传递数据，但注意HardExampleTrainer可能使用了混合数据
            if self.config.use_simulated_annealing:
                self.optimizer.step(p1, p2)
            else:
                self.optimizer.step()
            
            # 记录统计
            epoch_loss += total_loss.item()
            for k, v in batch_loss_dict.items():
                loss_stats[k] = loss_stats.get(k, 0.0) + v.item() if isinstance(v, torch.Tensor) else v

        self.scheduler.step()

        # 平均统计数据
        avg_loss = epoch_loss / num_batches
        avg_stats = {k: v / num_batches for k, v in loss_stats.items()}
        avg_stats['total_loss'] = avg_loss
        
        # 兼容父类日志打印所需的键名 (Key Mismatch Fix)
        if 'conflict' in avg_stats:
            avg_stats['conflict_loss'] = avg_stats['conflict']
        if 'entropy' in avg_stats:
            avg_stats['entropy_loss'] = avg_stats['entropy']
        
        # 记录难例库状态
        bank_stats = self.hard_example_bank.analyze()
        avg_stats['bank_size'] = bank_stats['size']
        avg_stats['bank_max_loss'] = bank_stats['max_loss']
        avg_stats['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        return avg_stats

    def add_geometric_hard_examples(self):
        """添加几何难例到银行"""
        try:
            # 生成 Moser Spindle
            points, edges = self.geometric_generator.generate_moser_spindle()
            
            # 将边转换为点对
            p1_list = []
            p2_list = []
            
            # points已经是float64 tensor
            points_np = points.detach().cpu().numpy()
            
            for i, j in edges:
                p1_list.append(points_np[i])
                p2_list.append(points_np[j])
            
            if p1_list:
                # 使用float64
                p1 = torch.tensor(np.stack(p1_list), dtype=torch.float64, device=self.device)
                p2 = torch.tensor(np.stack(p2_list), dtype=torch.float64, device=self.device)
                
                # 计算损失（给予较高初始损失，确保被选中）
                losses = torch.ones(len(p1_list), dtype=torch.float64, device=self.device) * 0.5
                
                self.hard_example_bank.add_batch(p1, p2, losses)
                print(f"Added {len(p1_list)} Moser Spindle edges to hard example bank")
        
        except Exception as e:
            print(f"Failed to add geometric hard examples: {e}")
            import traceback
            traceback.print_exc()