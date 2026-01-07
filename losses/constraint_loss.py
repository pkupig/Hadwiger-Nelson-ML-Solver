import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class ConstraintLoss(nn.Module):
    """Hadwiger-Nelson问题的约束损失"""
    
    def __init__(self, 
                 conflict_weight: float = 1.0,
                 entropy_weight: float = 0.1,
                 uniformity_weight: float = 0.01,
                 spectral_weight: float = 0.05,
                 temperature: float = 1.0):
        
        super().__init__()
        self.conflict_weight = conflict_weight
        self.entropy_weight = entropy_weight
        self.uniformity_weight = uniformity_weight
        self.spectral_weight = spectral_weight
        self.temperature = temperature
    
    def forward(self, 
                out1: torch.Tensor, 
                out2: torch.Tensor,
                all_outputs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        计算总损失
        
        Args:
            out1, out2: 距离为1的点对的颜色分布 [batch_size, num_colors]
            all_outputs: 所有点的颜色分布 [batch_size2, num_colors]，用于均匀性损失
            
        Returns:
            total_loss: 总损失
            loss_dict: 各个损失分量的字典
        """
        loss_dict = {}
        
        # 1. 冲突损失：距离为1的点应有不同的颜色
        conflict_loss = self.compute_conflict_loss(out1, out2)
        loss_dict['conflict'] = conflict_loss
        
        # 2. 熵损失：鼓励确定的颜色分配
        entropy_loss = self.compute_entropy_loss(torch.cat([out1, out2]))
        loss_dict['entropy'] = entropy_loss
        
        # 3. 均匀性损失：鼓励所有颜色被均匀使用（可选）
        if self.uniformity_weight > 0 and all_outputs is not None:
            uniformity_loss = self.compute_uniformity_loss(all_outputs)
            loss_dict['uniformity'] = uniformity_loss
        else:
            uniformity_loss = torch.tensor(0.0, device=out1.device)
        
        # 4. 谱损失：基于图拉普拉斯的正则化（可选）
        if self.spectral_weight > 0:
            spectral_loss = self.compute_spectral_loss(out1, out2)
            loss_dict['spectral'] = spectral_loss
        else:
            spectral_loss = torch.tensor(0.0, device=out1.device)
        
        # 总损失
        total_loss = (self.conflict_weight * conflict_loss +
                     self.entropy_weight * entropy_loss +
                     self.uniformity_weight * uniformity_loss +
                     self.spectral_weight * spectral_loss)
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict
    
    def compute_conflict_loss(self, out1: torch.Tensor, out2: torch.Tensor) -> torch.Tensor:
        """
        计算冲突损失：距离为1的点对颜色分布应正交
        
        Args:
            out1, out2: [batch_size, num_colors] 颜色概率分布
            
        Returns:
            conflict_loss: 标量损失
        """
        # 点积衡量两个分布的相似性
        # 如果两个点都倾向于同一种颜色，点积接近1
        dot_product = torch.sum(out1 * out2, dim=1)  # [batch_size]
        
        # 我们希望点积最小化（正交）
        conflict_loss = dot_product.mean()
        
        return conflict_loss
    
    def compute_entropy_loss(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        计算熵损失：鼓励确定的颜色分配（低熵）
        
        Args:
            outputs: [batch_size, num_colors] 颜色概率分布
            
        Returns:
            entropy_loss: 标量损失
        """
        # 计算香农熵
        log_probs = torch.log(outputs + 1e-10)
        entropy = -torch.sum(outputs * log_probs, dim=1)  # [batch_size]
        
        # 我们希望熵最小化（确定性高）
        entropy_loss = entropy.mean()
        
        return entropy_loss
    
    def compute_uniformity_loss(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        计算均匀性损失：鼓励所有颜色被均匀使用
        
        Args:
            outputs: [batch_size, num_colors] 颜色概率分布
            
        Returns:
            uniformity_loss: 标量损失
        """
        # 计算整个批次的平均颜色分布
        avg_distribution = outputs.mean(dim=0)  # [num_colors]
        
        # 理想均匀分布
        num_colors = outputs.size(1)
        target_distribution = torch.ones(num_colors, device=outputs.device) / num_colors
        
        # KL散度或MSE
        uniformity_loss = F.mse_loss(avg_distribution, target_distribution)
        
        return uniformity_loss
    
    def compute_spectral_loss(self, out1: torch.Tensor, out2: torch.Tensor) -> torch.Tensor:
        """
        计算谱损失：基于图拉普拉斯的正则化
        
        思想：在颜色空间构建图，鼓励平滑的颜色分配
        """
        batch_size = out1.size(0)
        
        # 构建邻接矩阵（简化的）
        # 对于每个点对，我们有一个边
        A = torch.zeros(batch_size * 2, batch_size * 2, device=out1.device)
        
        # 填充邻接矩阵（这里简化为仅考虑给定的点对）
        # 实际上应该构建完整的图，但为了效率我们简化
        
        # 简化的谱损失：鼓励相邻点颜色分布相似（但不同！）
        # 这是一个矛盾的目标，所以这个损失可能不是直接适用的
        # 我们暂时返回0，可以后续扩展
        
        return torch.tensor(0.0, device=out1.device)


class AdaptiveConstraintLoss(ConstraintLoss):
    """自适应权重的约束损失"""
    
    def __init__(self, 
                 initial_conflict_weight: float = 1.0,
                 initial_entropy_weight: float = 0.01,
                 weight_annealing: bool = True,
                 annealing_steps: int = 1000):
        
        super().__init__(conflict_weight=initial_conflict_weight,
                        entropy_weight=initial_entropy_weight)
        
        self.weight_annealing = weight_annealing
        self.annealing_steps = annealing_steps
        self.current_step = 0
    
    def update_weights(self, step: int):
        """根据训练步骤更新权重"""
        self.current_step = step
        
        if self.weight_annealing:
            # 逐渐增加熵损失的权重
            alpha = min(1.0, step / self.annealing_steps)
            self.entropy_weight = 0.01 + (0.1 - 0.01) * alpha
            
            # 可以逐渐减少冲突损失的权重
            # self.conflict_weight = 1.0 - 0.5 * alpha


class GeometricLoss(nn.Module):
    """几何感知的损失函数，考虑空间结构"""
    
    def __init__(self, dim: int = 2):
        super().__init__()
        self.dim = dim
    
    def compute_distance_weighted_loss(self, 
                                     out1: torch.Tensor, 
                                     out2: torch.Tensor,
                                     p1: torch.Tensor,
                                     p2: torch.Tensor) -> torch.Tensor:
        """
        根据点对距离加权损失
        
        Args:
            out1, out2: 颜色分布
            p1, p2: 坐标
            
        Returns:
            weighted_loss: 距离加权的损失
        """
        # 计算距离（应该是1，但可能有数值误差）
        distances = torch.norm(p1 - p2, dim=1)  # [batch_size]
        
        # 计算基础冲突损失
        dot_product = torch.sum(out1 * out2, dim=1)
        
        # 距离越接近1，权重越大
        # 使用高斯核
        sigma = 0.1
        weights = torch.exp(-(distances - 1.0)**2 / (2 * sigma**2))
        
        # 加权损失
        weighted_loss = torch.mean(weights * dot_product)
        
        return weighted_loss
    
    def compute_local_consistency_loss(self, 
                                     outputs: torch.Tensor,
                                     points: torch.Tensor,
                                     k: int = 5) -> torch.Tensor:
        """
        计算局部一致性损失：邻近的点应有相似的颜色分布
        
        Args:
            outputs: 颜色分布 [batch_size, num_colors]
            points: 坐标 [batch_size, dim]
            k: 最近邻数量
            
        Returns:
            consistency_loss: 一致性损失
        """
        batch_size = outputs.size(0)
        
        # 计算距离矩阵（对于大batch可能内存消耗大）
        if batch_size > 1000:  # 限制大小
            indices = torch.randperm(batch_size)[:1000]
            points = points[indices]
            outputs = outputs[indices]
            batch_size = 1000
        
        # 计算所有点对距离
        dist_matrix = torch.cdist(points, points)  # [batch_size, batch_size]
        
        # 找到k个最近邻（不包括自己）
        _, indices = torch.topk(dist_matrix, k=k+1, dim=1, largest=False)
        indices = indices[:, 1:]  # 去掉自己
        
        # 收集最近邻的输出
        neighbor_outputs = outputs[indices]  # [batch_size, k, num_colors]
        
        # 计算与最近邻的平均差异
        expanded_outputs = outputs.unsqueeze(1).expand(-1, k, -1)
        differences = torch.norm(expanded_outputs - neighbor_outputs, dim=2)
        
        consistency_loss = differences.mean()
        
        return consistency_loss