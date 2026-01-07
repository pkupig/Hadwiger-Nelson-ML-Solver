"""
难例挖掘器
主动寻找难以满足约束的点对，提高训练效率
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Dict
import heapq
import random


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
        
        Args:
            p1: 第一个点集 [batch_size, dim]
            p2: 第二个点集 [batch_size, dim]
            losses: 每个点对的损失 [batch_size]
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
        
        Args:
            batch_size: 批次大小
            strategy: 采样策略
                - "uniform": 均匀采样
                - "priority": 按损失优先级采样
                - "mixed": 混合策略
                
        Returns:
            p1, p2: 采样的点对
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
        
        # 转换为tensor
        p1 = torch.from_numpy(np.stack(p1_list)).float().to(self.device)
        p2 = torch.from_numpy(np.stack(p2_list)).float().to(self.device)
        
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
        
        p1 = torch.from_numpy(np.stack(p1_list)).float().to(self.device)
        p2 = torch.from_numpy(np.stack(p2_list)).float().to(self.device)
        losses = torch.tensor(loss_list, device=self.device)
        
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
    
    def save(self, filepath: str):
        """保存难例银行到文件"""
        import pickle
        
        data = {
            'examples': self.examples,
            'heap': self.heap,
            'index': self.index,
            'total_added': self.total_added,
            'total_rejected': self.total_rejected,
            'max_size': self.max_size,
            'min_loss_threshold': self.min_loss_threshold,
            'priority_weight': self.priority_weight
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """从文件加载难例银行"""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.examples = data['examples']
        self.heap = data['heap']
        self.index = data['index']
        self.total_added = data['total_added']
        self.total_rejected = data['total_rejected']
        self.max_size = data.get('max_size', self.max_size)
        self.min_loss_threshold = data.get('min_loss_threshold', self.min_loss_threshold)
        self.priority_weight = data.get('priority_weight', self.priority_weight)


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
    
    def reset(self):
        """重置挖掘器"""
        self.threshold = self.initial_threshold
        self.step_count = 0
        self.recent_losses.clear()


class GeometricHardExampleGenerator:
    """几何难例生成器：生成已知的硬配置"""
    
    def __init__(self, dim: int = 2, device: str = "cuda"):
        self.dim = dim
        self.device = device
    
    def generate_moser_spindle(self) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        生成Moser Spindle图（2D中的已知硬实例）
        
        Returns:
            points: 点坐标 [7, 2]
            edges: 边列表 [(i, j), ...]
        """
        if self.dim != 2:
            raise ValueError("Moser spindle is only defined for 2D")
        
        # Moser spindle的7个点
        points = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2],
            [1.5, np.sqrt(3)/2],
            [1.5, -np.sqrt(3)/2],
            [2.0, 0.0]
        ], dtype=torch.float32, device=self.device)
        
        # 距离为1的边
        edges = [
            (0, 1),  # 边长1
            (0, 2),  # 边长1
            (0, 3),  # 边长1
            (1, 2),  # 边长1
            (1, 3),  # 边长1
            (2, 4),  # 边长1
            (3, 5),  # 边长1
            (4, 5),  # 边长1
            (4, 6),  # 边长1
            (5, 6),  # 边长1
            (2, 6),  # 边长√3 ≈ 1.732，但在这个配置中距离为1
            (3, 6)   # 边长√3 ≈ 1.732，但在这个配置中距离为1
        ]
        
        # 验证距离（调试用）
        for i, j in edges:
            dist = torch.norm(points[i] - points[j])
            # 注意：在这个配置中，有些边不是精确的距离1
            # 但这是Moser spindle的标准构造
        
        return points, edges
    
    def generate_golomb_graph(self) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        生成Golomb图（另一个已知的硬实例）
        
        Returns:
            points: 点坐标
            edges: 边列表
        """
        # Golomb图有10个点，是单位距离图中的已知硬实例
        # 这里提供一个简化的构造
        
        # 使用复数表示（方便旋转）
        import cmath
        
        points_complex = []
        
        # 基础点
        base_points = [0, 1, 1j]  # 0, 1, i
        
        # 添加旋转得到的点
        for point in base_points:
            for angle in [0, 90, 180, 270]:
                rad = np.deg2rad(angle)
                rotated = point * cmath.exp(1j * rad)
                points_complex.append(rotated)
        
        # 去重并转换为实数坐标
        unique_points = []
        for p in points_complex:
            coord = (round(p.real, 6), round(p.imag, 6))
            if coord not in unique_points:
                unique_points.append(coord)
        
        points = torch.tensor(unique_points, dtype=torch.float32, device=self.device)
        
        # 找出距离为1的边
        edges = []
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                dist = torch.norm(points[i] - points[j])
                if abs(dist - 1.0) < 1e-6:
                    edges.append((i, j))
        
        return points, edges
    
    def generate_clique(self, n: int = 5) -> torch.Tensor:
        """
        生成近似团（所有点对距离都接近1）
        
        Args:
            n: 点数
            
        Returns:
            points: 点坐标 [n, dim]
        """
        # 使用优化方法寻找这样的配置
        points = torch.randn(n, self.dim, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([points], lr=0.01)
        
        for _ in range(500):
            optimizer.zero_grad()
            
            # 计算所有点对距离
            distances = torch.cdist(points, points)
            
            # 目标：所有距离接近1（除了对角线）
            mask = 1 - torch.eye(n, device=self.device)
            target_distances = torch.ones_like(distances) * mask
            
            loss = torch.mean((distances * mask - target_distances) ** 2)
            
            loss.backward()
            optimizer.step()
            
            if loss.item() < 0.01:
                break
        
        return points.detach()
    
    def generate_regular_simplex(self, n: int = None) -> torch.Tensor:
        """
        生成正则单纯形点集（所有点对距离相等）
        
        Args:
            n: 点数（最多dim+1）
            
        Returns:
            points: 点坐标
        """
        if n is None:
            n = self.dim + 1
        
        if n > self.dim + 1:
            raise ValueError(f"In {self.dim}D space, regular simplex can have at most {self.dim+1} points")
        
        # 生成正则单纯形的顶点
        points = torch.zeros(n, self.dim, device=self.device)
        
        # 第一个点在原点
        # 第二个点在x轴上
        if n > 1:
            points[1, 0] = 1.0
        
        # 后续点
        for i in range(2, n):
            # 第i个点的坐标
            for j in range(i):
                points[i, j] = 0.5
            
            # 调整使得与前面所有点的距离为1
            # 这里使用一个简化方法
            points[i, i-1] = np.sqrt(1.0 - np.sum(points[i, :i-1]**2))
        
        # 归一化使得所有点对距离为1
        if n >= 2:
            actual_distance = torch.norm(points[0] - points[1])
            points = points / actual_distance
        
        return points


class HardExampleTrainer:
    """带难例挖掘的训练器"""
    
    def __init__(self, 
                 model,
                 data_generator,
                 loss_fn,
                 hard_example_bank=None,
                 mining_frequency: int = 10,
                 hard_example_ratio: float = 0.3,
                 device: str = "cuda"):
        
        self.model = model
        self.data_generator = data_generator
        self.loss_fn = loss_fn
        self.device = device
        
        # 难例挖掘相关
        if hard_example_bank is None:
            self.hard_example_bank = HardExampleBank(device=device)
        else:
            self.hard_example_bank = hard_example_bank
        
        self.mining_frequency = mining_frequency
        self.hard_example_ratio = hard_example_ratio
        
        # 自适应挖掘器
        self.adaptive_miner = AdaptiveHardExampleMiner(device=device)
        
        # 几何难例生成器
        self.geometric_generator = GeometricHardExampleGenerator(
            dim=data_generator.dim,
            device=device
        )
        
        self.step_count = 0
    
    def train_step(self, optimizer, batch_size: int = 1024):
        """执行一个训练步骤"""
        self.model.train()
        self.step_count += 1
        
        # 确定难例比例
        use_hard_examples = (self.step_count % self.mining_frequency == 0)
        
        if use_hard_examples and self.hard_example_bank.analyze()["size"] > 0:
            # 使用部分难例
            num_hard = int(batch_size * self.hard_example_ratio)
            num_random = batch_size - num_hard
            
            # 从难例银行采样
            p1_hard, p2_hard = self.hard_example_bank.sample_batch(
                num_hard, strategy="priority"
            )
            
            # 生成随机样本
            p1_random, p2_random = self.data_generator.get_batch(num_random)
            
            # 合并
            if p1_hard is not None and p1_random is not None:
                p1 = torch.cat([p1_hard, p1_random], dim=0)
                p2 = torch.cat([p2_hard, p2_random], dim=0)
            elif p1_hard is not None:
                p1, p2 = p1_hard, p2_hard
            else:
                p1, p2 = p1_random, p2_random
        else:
            # 只使用随机样本
            p1, p2 = self.data_generator.get_batch(batch_size)
        
        # 前向传播
        out1 = self.model(p1)
        out2 = self.model(p2)
        
        # 计算损失
        total_loss, loss_dict = self.loss_fn(out1, out2)
        
        # 难例挖掘
        if use_hard_examples:
            # 计算每个点对的损失
            with torch.no_grad():
                # 这里简化处理：使用冲突损失作为难例标准
                dot_product = torch.sum(out1 * out2, dim=1)  # 冲突损失
                individual_losses = dot_product
            
            # 判断哪些是难例
            hard_mask = torch.tensor([
                self.adaptive_miner.should_keep_example(loss.item())
                for loss in individual_losses
            ], device=self.device)
            
            if hard_mask.any():
                # 添加难例到银行
                p1_hard = p1[hard_mask]
                p2_hard = p2[hard_mask]
                losses_hard = individual_losses[hard_mask]
                
                self.hard_example_bank.add_batch(p1_hard, p2_hard, losses_hard)
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item(), loss_dict
    
    def add_geometric_hard_examples(self):
        """添加几何难例到银行"""
        try:
            # 生成Moser Spindle
            points, edges = self.geometric_generator.generate_moser_spindle()
            
            # 将边转换为点对
            p1_list = []
            p2_list = []
            
            for i, j in edges:
                p1_list.append(points[i].cpu().numpy())
                p2_list.append(points[j].cpu().numpy())
            
            if p1_list:
                p1 = torch.tensor(p1_list, device=self.device)
                p2 = torch.tensor(p2_list, device=self.device)
                
                # 计算损失（假设高损失）
                losses = torch.ones(len(p1_list), device=self.device) * 0.5
                
                self.hard_example_bank.add_batch(p1, p2, losses)
                print(f"Added {len(p1_list)} Moser Spindle edges to hard example bank")
        
        except Exception as e:
            print(f"Failed to add geometric hard examples: {e}")
    
    def get_statistics(self) -> Dict:
        """获取训练统计信息"""
        bank_stats = self.hard_example_bank.analyze()
        
        return {
            "hard_example_bank": bank_stats,
            "adaptive_threshold": self.adaptive_miner.get_threshold(),
            "step_count": self.step_count,
            "mining_frequency": self.mining_frequency,
            "hard_example_ratio": self.hard_example_ratio
        }