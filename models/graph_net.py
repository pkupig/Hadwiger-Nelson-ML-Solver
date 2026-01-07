"""
图神经网络模型
将染色问题建模为图着色问题，使用GNN处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, Batch
from typing import Tuple, List, Optional, Dict
import numpy as np


class GeometricMessagePassing(MessagePassing):
    """几何感知的消息传递层"""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 edge_dim: int = 1):
        
        super().__init__(aggr='mean')
        
        # 消息函数
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )
        
        # 更新函数
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        """前向传播"""
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        """消息函数"""
        if edge_attr is not None:
            # 拼接源节点、目标节点和边特征
            msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            msg_input = torch.cat([x_i, x_j], dim=-1)
        
        return self.message_mlp(msg_input)
    
    def update(self, aggr_out, x):
        """更新函数"""
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)


class GeometricGraphConv(nn.Module):
    """几何图卷积层"""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 use_edge_features: bool = True):
        
        super().__init__()
        
        self.use_edge_features = use_edge_features
        
        # 节点特征变换
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 边特征变换（如果需要）
        if use_edge_features:
            self.edge_mlp = nn.Sequential(
                nn.Linear(1, 16),  # 距离作为边特征
                nn.ReLU(inplace=True),
                nn.Linear(16, 16)
            )
        
        # 消息传递层
        self.conv = GeometricMessagePassing(out_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr=None):
        """前向传播"""
        # 节点特征变换
        x = self.node_mlp(x)
        
        # 边特征变换
        if self.use_edge_features and edge_attr is not None:
            edge_attr = self.edge_mlp(edge_attr)
        
        # 图卷积
        x = self.conv(x, edge_index, edge_attr)
        
        return x


class GraphColorNet(nn.Module):
    """图神经网络颜色映射器"""
    
    def __init__(self, 
                 input_dim: int,
                 num_colors: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 conv_type: str = "geometric",
                 use_edge_features: bool = True,
                 dropout_rate: float = 0.1):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.num_colors = num_colors
        self.conv_type = conv_type
        self.use_edge_features = use_edge_features
        
        # 输入编码层
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 图卷积层
        self.conv_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if conv_type == "geometric":
                conv_layer = GeometricGraphConv(
                    hidden_dim, 
                    hidden_dim, 
                    use_edge_features
                )
            elif conv_type == "gcn":
                conv_layer = GCNConv(hidden_dim, hidden_dim)
            elif conv_type == "gat":
                conv_layer = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            elif conv_type == "sage":
                conv_layer = SAGEConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
            
            self.conv_layers.append(conv_layer)
        
        # 层归一化
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_colors)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, 
                data: Data,
                return_node_features: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: PyG Data对象，包含:
                - x: 节点特征 [num_nodes, input_dim]
                - edge_index: 边索引 [2, num_edges]
                - edge_attr: 边特征 [num_edges, 1]（可选）
                
        Returns:
            color_probs: 颜色概率 [num_nodes, num_colors]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        
        # 输入编码
        h = self.input_encoder(x)
        
        # 图卷积层
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norms)):
            # 残差连接
            residual = h
            
            # 图卷积
            if isinstance(conv, GeometricMessagePassing) or isinstance(conv, GeometricGraphConv):
                h = conv(h, edge_index, edge_attr)
            else:
                # 标准PyG卷积层
                h = conv(h, edge_index)
            
            # 层归一化和激活
            h = norm(h)
            h = F.relu(h, inplace=True)
            h = self.dropout(h)
            
            # 残差连接（除了第一层）
            if i > 0:
                h = h + residual
        
        # 保存节点特征（如果需要）
        node_features = h.clone()
        
        # 输出层
        logits = self.output_layer(h)
        color_probs = F.softmax(logits, dim=-1)
        
        if return_node_features:
            return color_probs, node_features
        else:
            return color_probs


class DynamicGraphBuilder:
    """动态图构建器：根据空间位置构建图"""
    
    def __init__(self, 
                 k_neighbors: int = 10,
                 max_distance: float = 2.0,
                 build_method: str = "knn"):
        
        self.k_neighbors = k_neighbors
        self.max_distance = max_distance
        self.build_method = build_method
    
    def build_graph(self, 
                    points: torch.Tensor,
                    unit_distance_edges: Optional[torch.Tensor] = None) -> Data:
        """
        从点集构建图
        
        Args:
            points: 点坐标 [num_points, dim]
            unit_distance_edges: 距离为1的边 [2, num_unit_edges]（可选）
            
        Returns:
            data: PyG Data对象
        """
        num_points = points.shape[0]
        
        # 节点特征（坐标）
        x = points
        
        # 构建边
        if self.build_method == "knn":
            # k最近邻图
            edge_index = self._build_knn_graph(points)
        elif self.build_method == "radius":
            # 半径图
            edge_index = self._build_radius_graph(points)
        elif self.build_method == "hybrid":
            # 混合：knn + 单位距离边
            edge_index = self._build_hybrid_graph(points, unit_distance_edges)
        else:
            raise ValueError(f"Unknown build_method: {self.build_method}")
        
        # 边特征（距离）
        edge_attr = self._compute_edge_attributes(points, edge_index)
        
        # 创建PyG Data对象
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data
    
    def _build_knn_graph(self, points: torch.Tensor) -> torch.Tensor:
        """构建k最近邻图"""
        from sklearn.neighbors import kneighbors_graph
        import scipy.sparse
        
        # 转换为numpy
        points_np = points.cpu().numpy()
        
        # 构建knn图
        adj_matrix = kneighbors_graph(
            points_np, 
            n_neighbors=self.k_neighbors,
            mode='connectivity',
            include_self=False
        )
        
        # 转换为边索引
        adj_coo = adj_matrix.tocoo()
        edge_index = torch.stack([
            torch.from_numpy(adj_coo.row).long(),
            torch.from_numpy(adj_coo.col).long()
        ])
        
        return edge_index
    
    def _build_radius_graph(self, points: torch.Tensor) -> torch.Tensor:
        """构建半径图"""
        from sklearn.neighbors import radius_neighbors_graph
        
        # 转换为numpy
        points_np = points.cpu().numpy()
        
        # 构建半径图
        adj_matrix = radius_neighbors_graph(
            points_np,
            radius=self.max_distance,
            mode='connectivity',
            include_self=False
        )
        
        # 转换为边索引
        adj_coo = adj_matrix.tocoo()
        edge_index = torch.stack([
            torch.from_numpy(adj_coo.row).long(),
            torch.from_numpy(adj_coo.col).long()
        ])
        
        return edge_index
    
    def _build_hybrid_graph(self, 
                           points: torch.Tensor,
                           unit_distance_edges: Optional[torch.Tensor]) -> torch.Tensor:
        """构建混合图：knn + 单位距离边"""
        
        # 基础knn图
        knn_edges = self._build_knn_graph(points)
        
        if unit_distance_edges is not None:
            # 合并边
            edge_index = torch.cat([knn_edges, unit_distance_edges], dim=1)
            
            # 去除重复边
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = knn_edges
        
        return edge_index
    
    def _compute_edge_attributes(self, 
                                points: torch.Tensor,
                                edge_index: torch.Tensor) -> torch.Tensor:
        """计算边特征（距离）"""
        src_nodes = points[edge_index[0]]
        dst_nodes = points[edge_index[1]]
        
        distances = torch.norm(src_nodes - dst_nodes, dim=1, keepdim=True)
        
        return distances


class GraphBasedColorMapper(nn.Module):
    """基于图神经网络的端到端颜色映射器"""
    
    def __init__(self, 
                 input_dim: int,
                 num_colors: int,
                 hidden_dim: int = 128,
                 num_gnn_layers: int = 3,
                 graph_builder_kwargs: Optional[Dict] = None):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.num_colors = num_colors
        
        # 图构建器
        if graph_builder_kwargs is None:
            graph_builder_kwargs = {
                'k_neighbors': 10,
                'max_distance': 2.0,
                'build_method': 'hybrid'
            }
        
        self.graph_builder = DynamicGraphBuilder(**graph_builder_kwargs)
        
        # 图神经网络
        self.gnn = GraphColorNet(
            input_dim=input_dim,
            num_colors=num_colors,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            conv_type="geometric",
            use_edge_features=True,
            dropout_rate=0.1
        )
    
    def forward(self, 
                points: torch.Tensor,
                unit_distance_edges: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            points: 点坐标 [batch_size, input_dim] 或 [num_points, input_dim]
            unit_distance_edges: 距离为1的边索引 [2, num_edges]（可选）
            
        Returns:
            color_probs: 颜色概率 [batch_size, num_colors]
        """
        # 构建图
        data = self.graph_builder.build_graph(points, unit_distance_edges)
        
        # 图神经网络前向传播
        color_probs = self.gnn(data)
        
        return color_probs
    
    def forward_with_graph(self, 
                          data: Data,
                          return_node_features: bool = False) -> torch.Tensor:
        """直接使用构建好的图进行前向传播"""
        return self.gnn(data, return_node_features=return_node_features)