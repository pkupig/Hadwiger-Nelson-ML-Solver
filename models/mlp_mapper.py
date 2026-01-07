import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

class FourierFeatures(nn.Module):
    """Fourier特征编码，用于更好地表示高频信息"""
    
    def __init__(self, input_dim: int, num_features: int = 256, sigma: float = 10.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        
        # 随机傅里叶特征
        self.B = nn.Parameter(torch.randn(input_dim, num_features // 2) * sigma, requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim]
        返回: [batch_size, 2 * num_features] (sin和cos)
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {x.shape[-1]}")
        
        # 投影
        proj = 2 * math.pi * x @ self.B  # [batch_size, num_features//2]
        
        # 计算sin和cos
        sin_features = torch.sin(proj)
        cos_features = torch.cos(proj)
        
        # 拼接
        return torch.cat([sin_features, cos_features], dim=-1)


class MLPColorMapper(nn.Module):
    """MLP颜色映射器：将坐标映射到颜色概率分布"""
    
    def __init__(self, 
                 input_dim: int, 
                 num_colors: int,
                 hidden_dims: List[int] = [128, 256, 128],
                 activation: str = "relu",
                 use_batch_norm: bool = True,
                 dropout_rate: float = 0.1,
                 use_fourier: bool = True,
                 fourier_features: int = 256,
                 fourier_sigma: float = 10.0):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.num_colors = num_colors
        self.use_fourier = use_fourier
        
        # Fourier特征编码
        if use_fourier:
            self.fourier = FourierFeatures(input_dim, fourier_features, fourier_sigma)
            mlp_input_dim = fourier_features  # sin+cos features
        else:
            self.fourier = None
            mlp_input_dim = input_dim
        
        # 构建MLP层
        layers = []
        prev_dim = mlp_input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation == "selu":
                layers.append(nn.SELU(inplace=True))
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            if dropout_rate > 0 and i < len(hidden_dims) - 1:  # 不在最后一层前加dropout
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_colors))
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        x: [batch_size, input_dim] 坐标
        返回: [batch_size, num_colors] 颜色概率分布
        """
        # Fourier特征编码
        if self.fourier is not None:
            x = self.fourier(x)
        
        # MLP前向传播
        logits = self.mlp(x)
        
        # 可调节温度的softmax
        if temperature != 1.0:
            logits = logits / temperature
        
        return F.softmax(logits, dim=-1)
    
    def get_color_assignment(self, x: torch.Tensor) -> torch.Tensor:
        """获取离散颜色分配（argmax）"""
        with torch.no_grad():
            probs = self.forward(x)
            return torch.argmax(probs, dim=-1)
    
    def get_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """计算输出的熵"""
        probs = self.forward(x)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.mean()


class ResidualColorMapper(nn.Module):
    """带残差连接的更深的MLP"""
    
    def __init__(self, input_dim: int, num_colors: int, num_blocks: int = 4, hidden_dim: int = 256):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 残差块
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            )
            self.blocks.append(block)
        
        self.output_layer = nn.Linear(hidden_dim, num_colors)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        
        for block in self.blocks:
            residual = h
            h = block(h)
            h = self.relu(h + residual)  # 残差连接
        
        logits = self.output_layer(h)
        return F.softmax(logits, dim=-1)