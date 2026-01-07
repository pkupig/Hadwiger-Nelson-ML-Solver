"""
几何工具函数
"""

import torch
import numpy as np
from typing import Tuple, List

def compute_unit_sphere_points(dim: int, num_points: int = 1000) -> torch.Tensor:
    """
    在单位超球面上生成均匀分布的点
    
    Args:
        dim: 维度
        num_points: 点数
        
    Returns:
        points: [num_points, dim] 单位球面上的点
    """
    # 使用正态分布然后归一化
    points = torch.randn(num_points, dim)
    points = points / torch.norm(points, dim=1, keepdim=True)
    return points

def find_unit_distance_pairs(points: torch.Tensor, 
                           tolerance: float = 1e-3) -> List[Tuple[int, int]]:
    """
    找到距离为1的点对
    
    Args:
        points: [n, dim] 点集
        tolerance: 距离容差
        
    Returns:
        edges: 距离在1±tolerance内的点对列表
    """
    n = len(points)
    edges = []
    
    # 计算距离矩阵（对小n可行）
    dist_matrix = torch.cdist(points, points)
    
    # 找到距离接近1的点对（不包括对角线）
    indices = torch.nonzero(
        (torch.abs(dist_matrix - 1.0) < tolerance) & 
        (torch.eye(n, device=points.device) == 0)
    )
    
    # 转换为列表并去重 (i,j) 和 (j,i)
    for i, j in indices:
        if i < j:  # 确保每个边只出现一次
            edges.append((i.item(), j.item()))
    
    return edges

def compute_chromatic_number_lower_bound(results: dict, 
                                       threshold: float = 1.0) -> dict:
    """
    根据实验结果计算染色数下界
    
    Args:
        results: 实验结果字典 {dim: {k: {violation_rate: float, ...}}}
        threshold: 可行阈值（冲突率%）
        
    Returns:
        lower_bounds: 每个维度的估计下界
    """
    lower_bounds = {}
    
    for dim, dim_results in results.items():
        feasible_k = []
        
        for k, result in dim_results.items():
            if result['final_violation_rate'] < threshold:
                feasible_k.append(k)
        
        if feasible_k:
            lower_bounds[dim] = min(feasible_k)
        else:
            lower_bounds[dim] = None
    
    return lower_bounds

def generate_regular_simplex(dim: int, scale: float = 1.0) -> torch.Tensor:
    """
    生成正则单纯形的顶点
    
    Args:
        dim: 维度
        scale: 缩放因子
        
    Returns:
        vertices: [dim+1, dim] 单纯形顶点
    """
    # 生成dim+1个点，使得任意两点距离相等
    vertices = torch.zeros(dim + 1, dim)
    
    # 前dim个点在坐标轴上
    for i in range(dim):
        vertices[i, i] = 1.0
    
    # 最后一个点
    vertices[dim, :] = (1 - np.sqrt(dim + 1)) / dim
    
    # 归一化使得所有点距离为scale
    current_dist = torch.norm(vertices[0] - vertices[1])
    vertices = vertices * (scale / current_dist)
    
    return vertices