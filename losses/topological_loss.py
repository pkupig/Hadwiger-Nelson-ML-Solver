import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TopologicalLoss(nn.Module):
    """拓扑损失：鼓励颜色分配保持拓扑性质"""
    
    def __init__(self, 
                 lambda_topo: float = 0.01,
                 persistence_threshold: float = 0.1):
        
        super().__init__()
        self.lambda_topo = lambda_topo
        self.persistence_threshold = persistence_threshold
    
    def compute_persistence(self, 
                          points: torch.Tensor,
                          values: torch.Tensor) -> List[Tuple[float, float]]:
        """
        计算持续性图（简化版）
        
        Args:
            points: [n, dim] 点集
            values: [n] 每个点的函数值（如颜色熵）
            
        Returns:
            persistence_pairs: 持续性对列表 [(birth, death), ...]
        """
        # 这是一个简化的实现，实际需要使用专门的拓扑库如gudhi
        # 这里我们模拟一个简单的版本
        
        n = len(points)
        if n < 3:
            return []
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(points, points).cpu().numpy()
        values_np = values.cpu().numpy()
        
        # 简化的持续性计算
        persistence_pairs = []
        
        # 找到局部极大值和极小值
        for i in range(1, n-1):
            if values_np[i] > values_np[i-1] and values_np[i] > values_np[i+1]:
                # 局部极大值
                birth = values_np[i]
                
                # 找到"死亡"值（周围的最小值）
                left_min = min(values_np[max(0, i-2):i])
                right_min = min(values_np[i+1:min(n, i+3)])
                death = min(left_min, right_min)
                
                persistence_pairs.append((birth, death))
        
        return persistence_pairs
    
    def compute_topological_loss(self, 
                               points: torch.Tensor,
                               outputs: torch.Tensor) -> torch.Tensor:
        """
        计算拓扑损失
        
        思想：鼓励颜色熵的拓扑结构保持稳定
        """
        # 计算每个点的颜色熵（不确定性）
        entropy = -torch.sum(outputs * torch.log(outputs + 1e-10), dim=1)
        
        # 计算持续性
        persistence_pairs = self.compute_persistence(points, entropy)
        
        if not persistence_pairs:
            return torch.tensor(0.0, device=points.device)
        
        # 计算持续性损失：鼓励长寿命的特征
        persistence_loss = 0.0
        for birth, death in persistence_pairs:
            persistence = birth - death
            if persistence < self.persistence_threshold:
                # 惩罚短寿命的特征
                persistence_loss += (self.persistence_threshold - persistence)
        
        if len(persistence_pairs) > 0:
            persistence_loss /= len(persistence_pairs)
        
        return torch.tensor(persistence_loss, device=points.device)
    
    def compute_winding_number_loss(self,
                                  points: torch.Tensor,
                                  colors: torch.Tensor,
                                  center: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算绕数损失：鼓励颜色围绕中心点的连续性
        
        Args:
            points: [n, dim] 点集
            colors: [n] 离散颜色标签
            center: 中心点坐标
            
        Returns:
            loss: 绕数损失
        """
        if points.shape[1] != 2:
            # 只支持2D
            return torch.tensor(0.0, device=points.device)
        
        n = len(points)
        if n < 3:
            return torch.tensor(0.0, device=points.device)
        
        if center is None:
            center = torch.mean(points, dim=0)
        
        # 计算角度
        vectors = points - center
        angles = torch.atan2(vectors[:, 1], vectors[:, 0])  # [-π, π]
        
        # 按角度排序
        sorted_indices = torch.argsort(angles)
        sorted_colors = colors[sorted_indices]
        
        # 计算颜色变化
        color_changes = torch.sum(sorted_colors[1:] != sorted_colors[:-1]).float()
        
        # 理想情况：颜色按角度连续变化，颜色变化次数最少
        # 这里我们鼓励颜色变化次数少
        return color_changes / n
    
    def compute_homology_loss(self,
                            points: torch.Tensor,
                            outputs: torch.Tensor,
                            num_colors: int) -> torch.Tensor:
        """
        计算同调损失：鼓励每个颜色连通分量的拓扑简单性
        """
        # 这是一个启发式的实现
        
        # 获取离散颜色分配
        color_ids = torch.argmax(outputs, dim=1)
        
        loss = 0.0
        
        for c in range(num_colors):
            # 该颜色的点
            mask = (color_ids == c)
            if torch.sum(mask) < 3:
                continue
            
            colored_points = points[mask]
            
            # 计算该颜色点集的凸包体积/面积
            if colored_points.shape[1] == 2:
                # 2D：计算凸包面积
                from scipy.spatial import ConvexHull
                try:
                    points_np = colored_points.cpu().detach().numpy()
                    hull = ConvexHull(points_np)
                    area = hull.volume  # 在2D中，volume是面积
                    
                    # 鼓励紧凑的形状（面积小）
                    # 但这是一个启发式，实际需要更复杂的拓扑分析
                    loss += area / len(colored_points)
                except:
                    pass
        
        return torch.tensor(loss, device=points.device) if loss > 0 else torch.tensor(0.0, device=points.device)
    
    def forward(self, 
                points: torch.Tensor,
                outputs: torch.Tensor,
                mode: str = "persistence") -> torch.Tensor:
        """
        计算拓扑损失
        
        Args:
            points: 坐标
            outputs: 颜色概率
            mode: 损失模式
            
        Returns:
            loss: 拓扑损失
        """
        if mode == "persistence":
            loss = self.compute_topological_loss(points, outputs)
        elif mode == "winding":
            colors = torch.argmax(outputs, dim=1)
            loss = self.compute_winding_number_loss(points, colors)
        elif mode == "homology":
            num_colors = outputs.shape[1]
            loss = self.compute_homology_loss(points, outputs, num_colors)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return self.lambda_topo * loss


class ContinuityLoss(nn.Module):
    """连续性损失：鼓励颜色分配的空间连续性"""
    
    def __init__(self, lambda_cont: float = 0.01, sigma: float = 1.0):
        super().__init__()
        self.lambda_cont = lambda_cont
        self.sigma = sigma
    
    def forward(self, 
                points: torch.Tensor,
                outputs: torch.Tensor) -> torch.Tensor:
        """
        计算连续性损失
        
        Args:
            points: [batch_size, dim] 坐标
            outputs: [batch_size, num_colors] 颜色概率
            
        Returns:
            loss: 连续性损失
        """
        batch_size = points.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=points.device)
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(points, points)
        
        # 高斯权重
        weights = torch.exp(-dist_matrix**2 / (2 * self.sigma**2))
        
        # 将对角线设为0
        weights = weights * (1 - torch.eye(batch_size, device=points.device))
        
        # 计算颜色分布的差异
        diff_matrix = torch.cdist(outputs, outputs, p=2)  # L2距离
        
        # 加权差异
        weighted_diff = weights * diff_matrix
        
        # 平均差异
        loss = torch.sum(weighted_diff) / (batch_size * (batch_size - 1))
        
        return self.lambda_cont * loss