"""
谱损失函数
基于图拉普拉斯算子的正则化损失，鼓励平滑的颜色分配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class SpectralLoss(nn.Module):
    """谱损失：基于图拉普拉斯的正则化"""
    
    def __init__(self, 
                 lambda_spec: float = 0.05,
                 k_neighbors: int = 10,
                 temperature: float = 0.1,
                 mode: str = "smoothness"):
        """
        Args:
            lambda_spec: 谱损失的权重
            k_neighbors: 最近邻数量（用于构建图）
            temperature: 注意力温度参数
            mode: 损失模式 ("smoothness", "consistency", "both")
        """
        super().__init__()
        
        self.lambda_spec = lambda_spec
        self.k_neighbors = k_neighbors
        self.temperature = temperature
        self.mode = mode
    
    def compute_graph_laplacian(self, 
                               points: torch.Tensor,
                               outputs: torch.Tensor,
                               epsilon: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算图拉普拉斯矩阵
        
        Args:
            points: [batch_size, dim] 坐标
            outputs: [batch_size, num_colors] 颜色概率
            epsilon: 数值稳定性常数
            
        Returns:
            L: 归一化拉普拉斯矩阵 [batch_size, batch_size]
            W: 邻接矩阵 [batch_size, batch_size]
        """
        batch_size = points.shape[0]
        
        # 对于大batch，限制大小以防止内存爆炸
        max_batch_for_laplacian = 1000
        if batch_size > max_batch_for_laplacian:
            indices = torch.randperm(batch_size)[:max_batch_for_laplacian]
            points = points[indices]
            outputs = outputs[indices]
            batch_size = max_batch_for_laplacian
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(points, points)  # [batch_size, batch_size]
        
        # 构建邻接矩阵（高斯核）
        # 使用自适应带宽
        sigma = torch.median(dist_matrix[dist_matrix > 0])
        W = torch.exp(-dist_matrix**2 / (2 * sigma**2 + epsilon))
        
        # 将对角线设为0
        W = W * (1 - torch.eye(batch_size, device=points.device))
        
        # 计算度矩阵
        D = torch.diag(torch.sum(W, dim=1))
        
        # 计算归一化拉普拉斯矩阵
        # L = I - D^(-1/2) W D^(-1/2)
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(torch.diag(D)) + epsilon))
        L = torch.eye(batch_size, device=points.device) - D_inv_sqrt @ W @ D_inv_sqrt
        
        return L, W
    
    def compute_manifold_smoothness(self, 
                                   points: torch.Tensor,
                                   outputs: torch.Tensor) -> torch.Tensor:
        """
        计算流形平滑度损失
        
        Args:
            points: [batch_size, dim] 坐标
            outputs: [batch_size, num_colors] 颜色概率
            
        Returns:
            loss: 平滑度损失标量
        """
        if points.shape[0] < 2:
            return torch.tensor(0.0, device=points.device)
        
        L, W = self.compute_graph_laplacian(points, outputs)
        
        # 平滑度：f^T L f
        # 对于每个颜色通道独立计算
        smoothness_per_channel = []
        num_colors = outputs.shape[1]
        
        for c in range(num_colors):
            f = outputs[:, c:c+1]  # [batch_size, 1]
            smoothness = torch.trace(f.T @ L @ f)
            smoothness_per_channel.append(smoothness)
        
        # 平均所有颜色通道
        total_smoothness = torch.stack(smoothness_per_channel).mean()
        
        # 归一化
        normalized_loss = total_smoothness / (points.shape[0] ** 2)
        
        return normalized_loss
    
    def compute_local_consistency(self, 
                                 points: torch.Tensor,
                                 outputs: torch.Tensor) -> torch.Tensor:
        """
        计算局部一致性损失
        
        Args:
            points: [batch_size, dim] 坐标
            outputs: [batch_size, num_colors] 颜色概率
            
        Returns:
            loss: 局部一致性损失
        """
        batch_size = points.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=points.device)
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(points, points)  # [batch_size, batch_size]
        
        # 将对角线设为无穷大，避免选择自己
        dist_matrix = dist_matrix + torch.diag(torch.full((batch_size,), float('inf'), device=points.device))
        
        # 获取k个最近邻的索引
        k = min(self.k_neighbors, batch_size - 1)
        _, indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)  # [batch_size, k]
        
        # 收集邻居的输出
        neighbor_outputs = outputs[indices]  # [batch_size, k, num_colors]
        
        # 计算与邻居的平均差异
        expanded_outputs = outputs.unsqueeze(1).expand_as(neighbor_outputs)
        differences = torch.norm(expanded_outputs - neighbor_outputs, dim=2)  # [batch_size, k]
        
        # 加权平均（距离越近，权重越大）
        neighbor_dists = torch.gather(dist_matrix, 1, indices)
        weights = torch.exp(-neighbor_dists / self.temperature)
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
        
        weighted_differences = torch.sum(weights * differences, dim=1)
        
        return weighted_differences.mean()
    
    def compute_spectral_clustering_loss(self,
                                        points: torch.Tensor,
                                        outputs: torch.Tensor) -> torch.Tensor:
        """
        计算谱聚类损失：鼓励不同簇之间的分离
        
        Args:
            points: 坐标
            outputs: 颜色概率
            
        Returns:
            loss: 谱聚类损失
        """
        if points.shape[0] < 2:
            return torch.tensor(0.0, device=points.device)
        
        # 计算拉普拉斯矩阵
        L, W = self.compute_graph_laplacian(points, outputs)
        
        # 计算瑞利商：f^T L f / f^T f
        num_colors = outputs.shape[1]
        
        rayleigh_quotients = []
        for c in range(num_colors):
            f = outputs[:, c:c+1]  # [batch_size, 1]
            numerator = torch.trace(f.T @ L @ f)
            denominator = torch.trace(f.T @ f)
            
            if denominator > 1e-8:
                rayleigh_quotients.append(numerator / denominator)
        
        if rayleigh_quotients:
            # 我们希望瑞利商最小化（在同一个簇内平滑）
            loss = torch.stack(rayleigh_quotients).mean()
        else:
            loss = torch.tensor(0.0, device=points.device)
        
        return loss
    
    def compute_normalized_cut_loss(self,
                                  points: torch.Tensor,
                                  outputs: torch.Tensor) -> torch.Tensor:
        """
        计算归一化割损失：鼓励平衡的聚类
        
        Args:
            points: 坐标
            outputs: 颜色概率
            
        Returns:
            loss: 归一化割损失
        """
        batch_size = points.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=points.device)
        
        # 计算邻接矩阵
        dist_matrix = torch.cdist(points, points)
        sigma = torch.median(dist_matrix[dist_matrix > 0])
        W = torch.exp(-dist_matrix**2 / (2 * sigma**2))
        W = W * (1 - torch.eye(batch_size, device=points.device))
        
        # 计算度矩阵
        D = torch.diag(torch.sum(W, dim=1))
        
        num_colors = outputs.shape[1]
        ncut_loss = 0.0
        
        for c in range(num_colors):
            # 获取该颜色的"隶属度"
            f = outputs[:, c]  # [batch_size]
            
            # 计算指示向量（软分配）
            indicator = f
            
            # 归一化割：cut(A, Ā) / vol(A)
            # 其中cut(A, Ā) = Σ_{i∈A, j∉A} W_{ij}
            # vol(A) = Σ_{i∈A} D_{ii}
            
            # 计算cut
            cut = torch.sum(W * (indicator.view(-1, 1) * (1 - indicator.view(1, -1))))
            
            # 计算vol
            vol = torch.sum(D * indicator)
            
            if vol > 1e-8:
                ncut = cut / vol
                ncut_loss += ncut
        
        # 平均所有颜色
        if num_colors > 0:
            ncut_loss = ncut_loss / num_colors
        
        return ncut_loss
    
    def forward(self, 
                points: torch.Tensor,
                outputs: torch.Tensor,
                mode: Optional[str] = None) -> torch.Tensor:
        """
        计算谱损失
        
        Args:
            points: 坐标
            outputs: 颜色概率
            mode: 损失模式（如果为None，使用self.mode）
            
        Returns:
            loss: 谱损失
        """
        if mode is None:
            mode = self.mode
        
        if mode == "smoothness":
            loss = self.compute_manifold_smoothness(points, outputs)
        elif mode == "consistency":
            loss = self.compute_local_consistency(points, outputs)
        elif mode == "spectral_clustering":
            loss = self.compute_spectral_clustering_loss(points, outputs)
        elif mode == "normalized_cut":
            loss = self.compute_normalized_cut_loss(points, outputs)
        elif mode == "both":
            loss1 = self.compute_manifold_smoothness(points, outputs)
            loss2 = self.compute_local_consistency(points, outputs)
            loss = (loss1 + loss2) / 2
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return self.lambda_spec * loss


class GraphCutLoss(nn.Module):
    """图割损失：鼓励在图的割集上颜色平滑过渡"""
    
    def __init__(self, lambda_cut: float = 0.01, temperature: float = 0.1):
        super().__init__()
        self.lambda_cut = lambda_cut
        self.temperature = temperature
    
    def forward(self, 
                points1: torch.Tensor,
                points2: torch.Tensor,
                outputs1: torch.Tensor,
                outputs2: torch.Tensor) -> torch.Tensor:
        """
        计算图割损失
        
        Args:
            points1, points2: 点对坐标 [batch_size, dim]
            outputs1, outputs2: 对应的颜色概率 [batch_size, num_colors]
            
        Returns:
            loss: 图割损失
        """
        # 计算点对之间的距离
        distances = torch.norm(points1 - points2, dim=1, keepdim=True)  # [batch_size, 1]
        
        # 计算颜色分布差异
        # 使用对称KL散度（Jensen-Shannon散度）
        p = F.softmax(outputs1 / self.temperature, dim=-1)
        q = F.softmax(outputs2 / self.temperature, dim=-1)
        
        m = 0.5 * (p + q)
        
        kl_pm = F.kl_div(F.log_softmax(outputs1 / self.temperature, dim=-1), 
                        m, reduction='batchmean')
        kl_qm = F.kl_div(F.log_softmax(outputs2 / self.temperature, dim=-1), 
                        m, reduction='batchmean')
        
        js_divergence = 0.5 * (kl_pm + kl_qm)
        
        # 距离越近，颜色差异应该越小（加权）
        # 但我们希望距离为1的点对颜色不同，所以这是一个微妙的问题
        # 这里我们使用一个权衡：鼓励适度的差异（既不是完全相同，也不是完全不同）
        target_divergence = 0.5  # 目标JS散度
        
        # 损失：鼓励JS散度接近目标值
        loss = torch.mean((js_divergence - target_divergence) ** 2)
        
        return self.lambda_cut * loss


class LaplacianEigenLoss(nn.Module):
    """拉普拉斯特征损失：鼓励颜色分配对齐于图的低维特征向量"""
    
    def __init__(self, lambda_eigen: float = 0.02, num_eigenvectors: int = 3):
        super().__init__()
        self.lambda_eigen = lambda_eigen
        self.num_eigenvectors = num_eigenvectors
    
    def compute_laplacian_eigenvectors(self, 
                                      points: torch.Tensor,
                                      k_neighbors: int = 10) -> torch.Tensor:
        """
        计算图拉普拉斯的特征向量
        
        Args:
            points: 坐标 [batch_size, dim]
            k_neighbors: 最近邻数量
            
        Returns:
            eigenvectors: 前k个特征向量 [batch_size, num_eigenvectors]
        """
        batch_size = points.shape[0]
        
        if batch_size < self.num_eigenvectors + 1:
            # 返回随机向量
            return torch.randn(batch_size, self.num_eigenvectors, device=points.device)
        
        # 构建k最近邻图
        from sklearn.neighbors import kneighbors_graph
        import scipy.sparse.linalg as sla
        
        points_np = points.cpu().numpy()
        
        # 构建邻接矩阵
        W = kneighbors_graph(points_np, n_neighbors=k_neighbors, mode='connectivity', include_self=False)
        W = 0.5 * (W + W.T)  # 对称化
        
        # 计算度矩阵
        D = np.diag(W.sum(axis=1).A1)
        
        # 计算拉普拉斯矩阵
        L = D - W
        
        # 计算特征值和特征向量
        try:
            eigenvalues, eigenvectors = sla.eigsh(L, k=self.num_eigenvectors, which='SM')
            eigenvectors = torch.from_numpy(eigenvectors).float().to(points.device)
        except:
            # 如果失败，返回随机向量
            eigenvectors = torch.randn(batch_size, self.num_eigenvectors, device=points.device)
        
        return eigenvectors
    
    def forward(self, 
                points: torch.Tensor,
                outputs: torch.Tensor) -> torch.Tensor:
        """
        计算特征损失
        
        Args:
            points: 坐标
            outputs: 颜色概率
            
        Returns:
            loss: 特征损失
        """
        # 计算特征向量
        eigenvectors = self.compute_laplacian_eigenvectors(points)
        
        # 我们希望颜色分配与特征向量对齐
        # 即，颜色概率应该在特征向量张成的子空间中
        
        num_colors = outputs.shape[1]
        loss = 0.0
        
        for c in range(num_colors):
            color_vector = outputs[:, c]  # [batch_size]
            
            # 投影到特征向量空间
            projections = []
            for i in range(self.num_eigenvectors):
                eigenvector = eigenvectors[:, i]
                projection = torch.dot(color_vector, eigenvector)
                projections.append(projection ** 2)
            
            # 计算重构误差
            # 将颜色向量重构为特征向量的线性组合
            reconstructed = torch.zeros_like(color_vector)
            for i in range(self.num_eigenvectors):
                coeff = torch.dot(color_vector, eigenvectors[:, i])
                reconstructed = reconstructed + coeff * eigenvectors[:, i]
            
            reconstruction_error = torch.norm(color_vector - reconstructed) ** 2
            loss += reconstruction_error
        
        # 平均所有颜色
        if num_colors > 0:
            loss = loss / num_colors
        
        return self.lambda_eigen * loss


class CombinedSpectralLoss(nn.Module):
    """组合谱损失：多个谱损失的加权和"""
    
    def __init__(self, 
                 loss_weights: dict = None,
                 **kwargs):
        super().__init__()
        
        if loss_weights is None:
            loss_weights = {
                'smoothness': 1.0,
                'consistency': 0.5,
                'graph_cut': 0.2,
                'eigen': 0.1
            }
        
        self.loss_weights = loss_weights
        
        # 初始化各个损失
        self.smoothness_loss = SpectralLoss(
            lambda_spec=1.0,
            mode='smoothness',
            **kwargs
        )
        
        self.consistency_loss = SpectralLoss(
            lambda_spec=1.0,
            mode='consistency',
            **kwargs
        )
        
        self.graph_cut_loss = GraphCutLoss(
            lambda_cut=1.0,
            **kwargs
        )
        
        self.eigen_loss = LaplacianEigenLoss(
            lambda_eigen=1.0,
            **kwargs
        )
    
    def forward(self, 
                points: torch.Tensor,
                outputs: torch.Tensor,
                points1: Optional[torch.Tensor] = None,
                points2: Optional[torch.Tensor] = None,
                outputs1: Optional[torch.Tensor] = None,
                outputs2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算组合谱损失
        
        Args:
            points: 单点坐标（用于smoothness和consistency）
            outputs: 单点颜色概率
            points1, points2: 点对坐标（用于graph_cut）
            outputs1, outputs2: 点对颜色概率
            
        Returns:
            total_loss: 总损失
            loss_dict: 各个损失分量的字典
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 平滑度损失
        if 'smoothness' in self.loss_weights and self.loss_weights['smoothness'] > 0:
            smoothness = self.smoothness_loss(points, outputs)
            weighted_smoothness = self.loss_weights['smoothness'] * smoothness
            loss_dict['smoothness'] = weighted_smoothness.item()
            total_loss += weighted_smoothness
        
        # 一致性损失
        if 'consistency' in self.loss_weights and self.loss_weights['consistency'] > 0:
            consistency = self.consistency_loss(points, outputs)
            weighted_consistency = self.loss_weights['consistency'] * consistency
            loss_dict['consistency'] = weighted_consistency.item()
            total_loss += weighted_consistency
        
        # 图割损失（需要点对）
        if ('graph_cut' in self.loss_weights and self.loss_weights['graph_cut'] > 0 and
            points1 is not None and points2 is not None and
            outputs1 is not None and outputs2 is not None):
            graph_cut = self.graph_cut_loss(points1, points2, outputs1, outputs2)
            weighted_graph_cut = self.loss_weights['graph_cut'] * graph_cut
            loss_dict['graph_cut'] = weighted_graph_cut.item()
            total_loss += weighted_graph_cut
        
        # 特征损失
        if 'eigen' in self.loss_weights and self.loss_weights['eigen'] > 0:
            eigen = self.eigen_loss(points, outputs)
            weighted_eigen = self.loss_weights['eigen'] * eigen
            loss_dict['eigen'] = weighted_eigen.item()
            total_loss += weighted_eigen
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict