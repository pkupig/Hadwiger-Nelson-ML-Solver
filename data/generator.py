import torch
import numpy as np
from typing import Tuple, List, Optional
import random

class PointPairGenerator:
    """生成满足距离约束的点对"""
    
    def __init__(self, 
                 dim: int = 2,
                 batch_size: int = 4096,
                 space_range: Tuple[float, float] = (-5.0, 5.0),
                 device: str = "cuda",
                 seed: Optional[int] = None):
        
        self.dim = dim
        self.batch_size = batch_size
        self.space_range = space_range
        self.device = device
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def get_batch(self, strategy: str = "mixed_chain") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成一批距离为1的点对
        
        Args:
            strategy: 采样策略
                - "unit_sphere": 在单位超球面上采样
                - "random_distance": 随机距离但固定一对为1
                - "grid": 网格采样（用于验证）
        """
        if strategy == "mixed_chain":
            return self._sample_chain_structure()
        elif strategy == "unit_sphere":
            return self._sample_unit_sphere()
        elif strategy == "random_distance":
            return self._sample_random_distance()
        elif strategy == "grid":
            return self._sample_grid()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _sample_chain_structure(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        混合生成独立对和链式结构 (A-B-C)
        链式结构强迫模型处理局部依赖：若 A=Red, B!=Red; 若 B=Green, C!=Green。
        """
        # 一半数据生成标准点对 A-B
        half_batch = self.batch_size // 2
        p1_a, p2_a = self._sample_unit_sphere_subset(half_batch)
        
        # 另一半数据生成链式 B-C (利用上一组的 p2 作为起点)
        # 这样我们在同一个 batch 里有了 A->B 和 B->C (虽然在 Tensor 中是分开的，
        # 但如果是基于 Graph 的训练或后续 Hard Mining 会捕捉到这种空间关系)
        # 更直接的方法是生成 explicit chains:
        
        # 重新采样起始点
        start_points = torch.rand(half_batch, self.dim, device=self.device) * (self.space_range[1] - self.space_range[0]) + self.space_range[0]
        
        # A -> B
        v1 = torch.randn(half_batch, self.dim, device=self.device)
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
        points_B = start_points + v1
        
        # B -> C
        v2 = torch.randn(half_batch, self.dim, device=self.device)
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
        points_C = points_B + v2
        
        # 将 (A, B) 和 (B, C) 拼接返回
        p1 = torch.cat([start_points, points_B], dim=0)
        p2 = torch.cat([points_B, points_C], dim=0)
        
        return p1, p2

    def _sample_unit_sphere(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """在单位超球面上均匀采样位移向量"""
        # 1. 随机采样起始点 P1
        low, high = self.space_range
        p1 = torch.rand(self.batch_size, self.dim, device=self.device) * (high - low) + low
        
        # 2. 生成单位随机向量 v（超球面上的均匀分布）
        # 使用正态分布然后归一化
        v = torch.randn(self.batch_size, self.dim, device=self.device)
        v_norm = torch.norm(v, dim=1, keepdim=True)
        v = v / (v_norm + 1e-8)
        
        # 3. 确保精确的单位长度
        v = v / torch.norm(v, dim=1, keepdim=True)
        
        # 4. 得到 P2 = P1 + v
        p2 = p1 + v
        
        # 验证距离（调试用）
        distances = torch.norm(p2 - p1, dim=1)
        assert torch.allclose(distances, torch.ones_like(distances), rtol=1e-8), \
            f"Distance constraint violated! Mean distance: {distances.mean().item()}"
        
        return p1, p2
    
    def _sample_unit_sphere_subset(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成 n 个单位距离的点对 (辅助函数)
        
        Args:
            n: 生成的点对数量
            
        Returns:
            p1, p2: [n, dim]
        """
        # 1. 随机采样起始点 P1
        low, high = self.space_range
        p1 = torch.rand(n, self.dim, device=self.device) * (high - low) + low
        
        # 2. 生成单位随机向量 v（超球面上的均匀分布）
        # 使用正态分布然后归一化
        v = torch.randn(n, self.dim, device=self.device)
        v_norm = torch.norm(v, dim=1, keepdim=True)
        # 防止除零
        v = v / (v_norm + 1e-8)
        
        # 3. 确保精确的单位长度 (Double check normalization)
        # 建议：如果开启了 float64 精度，这里会更精确
        v = v / torch.norm(v, dim=1, keepdim=True)
        
        # 4. 得到 P2 = P1 + v
        p2 = p1 + v
        
        # 验证距离
        # if self.device == 'cpu' or n < 10000:
        #     distances = torch.norm(p2 - p1, dim=1)
        #     # 注意：使用 float32，rtol应该是 1e-5；如果 float64，可以是 1e-7
        #     assert torch.allclose(distances, torch.ones_like(distances), rtol=1e-5), \
        #         f"Distance constraint violated! Mean: {distances.mean().item()}"
        
        return p1, p2

    def _sample_random_distance(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样随机距离的点对，但确保至少一对距离为1"""
        low, high = self.space_range
        p1 = torch.rand(self.batch_size, self.dim, device=self.device) * (high - low) + low
        
        # 随机方向
        directions = torch.randn(self.batch_size, self.dim, device=self.device)
        directions = directions / (torch.norm(directions, dim=1, keepdim=True) + 1e-8)
        
        # 随机距离，但确保第一对的距离为1
        distances = torch.rand(self.batch_size, 1, device=self.device) * 2.0 + 0.1  # [0.1, 2.1]
        distances[0] = 1.0  # 确保至少有一对距离为1
        
        p2 = p1 + directions * distances
        
        return p1, p2
    
    def _sample_grid(self, resolution: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """在网格上采样点对（用于验证）"""
        # 创建网格
        x = torch.linspace(self.space_range[0], self.space_range[1], resolution, device=self.device)
        coords = torch.meshgrid(*[x for _ in range(self.dim)], indexing='ij')
        grid_points = torch.stack([c.flatten() for c in coords], dim=1)
        
        # 随机选择点对
        indices = torch.randint(0, len(grid_points), (self.batch_size, 2), device=self.device)
        p1 = grid_points[indices[:, 0]]
        p2 = grid_points[indices[:, 1]]
        
        return p1, p2
    
    def generate_moser_spindle(self) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        生成Moser Spindle图（2D中的已知硬实例）
        返回：点和边列表
        """
        if self.dim != 2:
            raise ValueError("Moser spindle is only defined for 2D")
        
        # Moser spindle的7个点
        points = torch.tensor([
            [0, 0],
            [1, 0],
            [0.5, np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2],
            [1.5, np.sqrt(3)/2],
            [1.5, -np.sqrt(3)/2],
            [2, 0]
        ], dtype=torch.float32, device=self.device)
        
        # 距离为1的边
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 4),
            (3, 5), (4, 5), (4, 6),
            (5, 6), (2, 6), (3, 6)
        ]
        
        return points, edges
    
    def generate_hard_configuration(self, n_points: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成已知的硬配置点集
        """
        points = torch.randn(n_points, self.dim, device=self.device)
        points = points / torch.norm(points, dim=1, keepdim=True) * 2.0
        
        # 找到所有距离接近1的点对
        distances = torch.cdist(points, points)
        mask = (torch.abs(distances - 1.0) < 0.1) & (torch.eye(n_points, device=self.device) == 0)
        
        # 随机选择一些边
        indices = torch.nonzero(mask)
        if len(indices) > 0:
            selected = indices[torch.randperm(len(indices))[:self.batch_size]]
            p1 = points[selected[:, 0]]
            p2 = points[selected[:, 1]]
            return p1, p2
        
        # 如果没有找到，返回随机点对
        return self.get_batch()


class HardExampleMiner:
    """难例挖掘器：找到最难满足约束的点对"""
    
    def __init__(self, memory_size: int = 50000, top_k_percentile: float = 5.0):
        self.memory_size = memory_size
        self.top_k_percentile = top_k_percentile
        self.p1_memory = []
        self.p2_memory = []
        self.loss_memory = []
    
    def add_examples(self, p1: torch.Tensor, p2: torch.Tensor, losses: torch.Tensor):
        """添加新的例子到内存"""
        p1_np = p1.cpu().numpy()
        p2_np = p2.cpu().numpy()
        losses_np = losses.cpu().numpy()
        
        self.p1_memory.append(p1_np)
        self.p2_memory.append(p2_np)
        self.loss_memory.append(losses_np)
        
        # 限制内存大小
        if len(self.p1_memory) * p1_np.shape[0] > self.memory_size:
            self.p1_memory = self.p1_memory[-self.memory_size // p1_np.shape[0]:]
            self.p2_memory = self.p2_memory[-self.memory_size // p2_np.shape[0]:]
            self.loss_memory = self.loss_memory[-self.memory_size // losses_np.shape[0]:]
    
    def get_hard_examples(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取最难的例子"""
        if not self.p1_memory:
            return None, None
        
        # 合并所有记忆
        all_p1 = np.concatenate(self.p1_memory, axis=0)
        all_p2 = np.concatenate(self.p2_memory, axis=0)
        all_losses = np.concatenate(self.loss_memory, axis=0)
        
        # 选择loss最高的top_k_percentile%
        threshold = np.percentile(all_losses, 100 - self.top_k_percentile)
        hard_indices = np.where(all_losses >= threshold)[0]
        
        if len(hard_indices) == 0:
            return None, None
        
        # 随机选择batch_size个hard examples
        selected = np.random.choice(hard_indices, size=min(batch_size, len(hard_indices)), replace=False)
        
        p1 = torch.from_numpy(all_p1[selected]).float()
        p2 = torch.from_numpy(all_p2[selected]).float()
        
        return p1, p2