"""
注意力机制增强的颜色映射器
结合局部注意力来捕捉空间依赖关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional
import numpy as np


class SpatialAttention(nn.Module):
    """空间注意力模块：学习哪些空间位置是重要的"""
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, channels, length]
        返回: [batch_size, channels, length] 注意力权重
        """
        batch_size, channels, length = x.size()
        
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x).view(batch_size, channels))
        max_out = self.fc(self.max_pool(x).view(batch_size, channels))
        
        channel_weights = self.sigmoid(avg_out + max_out).view(batch_size, channels, 1)
        
        # 应用通道注意力
        x_weighted = x * channel_weights.expand_as(x)
        
        return x_weighted


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 查询、键、值的线性变换
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch_size, seq_len, embed_dim]
        返回: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # 线性变换并分头
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用掩码（如果需要）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # 注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, v)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class PositionalEncoding2D(nn.Module):
    """2D位置编码（类似Transformer，但适应空间坐标）"""
    
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        
        self.dim = dim
        
        # 创建位置编码
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 使用不同的频率
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * 
            (-math.log(10000.0) / dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, dim]
        返回: [batch_size, seq_len, dim] 带位置编码
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class AttentionColorMapper(nn.Module):
    """注意力机制增强的颜色映射器"""
    
    def __init__(self, 
                 input_dim: int,
                 num_colors: int,
                 hidden_dim: int = 256,
                 num_attention_heads: int = 4,
                 num_attention_layers: int = 2,
                 num_mlp_layers: int = 2,
                 dropout_rate: float = 0.1,
                 use_fourier: bool = True,
                 fourier_features: int = 256):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.num_colors = num_colors
        self.use_fourier = use_fourier
        
        # Fourier特征编码
        if use_fourier:
            from models.fourier_features import FourierFeatures
            self.fourier = FourierFeatures(input_dim, fourier_features)
            mlp_input_dim = fourier_features
        else:
            self.fourier = None
            mlp_input_dim = input_dim
        
        # 输入投影层
        self.input_proj = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 注意力层
        self.attention_layers = nn.ModuleList()
        for _ in range(num_attention_layers):
            attention_layer = nn.ModuleDict({
                'attn': MultiHeadSelfAttention(hidden_dim, num_attention_heads, dropout_rate),
                'norm1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                'norm2': nn.LayerNorm(hidden_dim),
                'dropout': nn.Dropout(dropout_rate)
            })
            self.attention_layers.append(attention_layer)
        
        # MLP层（注意力之后）
        self.mlp_layers = nn.ModuleList()
        for i in range(num_mlp_layers):
            self.mlp_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate)
                )
            )
        
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
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, 
                x: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor:
        """
        x: [batch_size, input_dim] 坐标
        返回: [batch_size, num_colors] 颜色概率
        """
        batch_size = x.shape[0]
        
        # Fourier特征编码
        if self.fourier is not None:
            x = self.fourier(x)
        
        # 输入投影
        h = self.input_proj(x)
        
        # 添加序列维度（为了注意力机制）
        # 我们将每个点视为序列中的一个元素
        h_seq = h.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        attention_weights = []
        
        # 注意力层
        for layer in self.attention_layers:
            # 残差连接和层归一化
            residual = h_seq
            
            # 自注意力
            attn_output, attn = layer['attn'](h_seq)
            attention_weights.append(attn)
            
            # Dropout和残差连接
            h_seq = layer['norm1'](h_seq + layer['dropout'](attn_output))
            
            # Feed-Forward Network
            ffn_output = layer['ffn'](h_seq)
            h_seq = layer['norm2'](h_seq + layer['dropout'](ffn_output))
        
        # 移除序列维度
        h = h_seq.squeeze(1)  # [batch_size, hidden_dim]
        
        # MLP层
        for mlp_layer in self.mlp_layers:
            h = mlp_layer(h)
        
        # 输出
        logits = self.output_layer(h)
        probs = F.softmax(logits, dim=-1)
        
        if return_attention:
            return probs, attention_weights
        else:
            return probs
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """获取注意力权重图"""
        with torch.no_grad():
            _, attention_weights = self.forward(x, return_attention=True)
        return attention_weights


class LocalAttentionColorMapper(nn.Module):
    """局部注意力颜色映射器：考虑空间邻域的注意力"""
    
    def __init__(self, 
                 input_dim: int,
                 num_colors: int,
                 hidden_dim: int = 256,
                 num_neighbors: int = 8,
                 dropout_rate: float = 0.1):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.num_colors = num_colors
        self.num_neighbors = num_neighbors
        
        # 输入编码
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 局部注意力模块
        self.local_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_colors)
        )
    
    def forward(self, 
                x: torch.Tensor,
                neighbors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch_size, input_dim] 中心点坐标
        neighbors: [batch_size, num_neighbors, input_dim] 邻居点坐标（可选）
        返回: [batch_size, num_colors] 颜色概率
        """
        batch_size = x.shape[0]
        
        # 如果未提供邻居，则从批量中随机选择
        if neighbors is None:
            # 随机选择邻居（实际应用中应根据空间距离选择）
            indices = torch.randint(0, batch_size, (batch_size, self.num_neighbors))
            neighbors = x[indices]  # [batch_size, num_neighbors, input_dim]
        
        # 编码中心点
        center_encoded = self.input_encoder(x).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 编码邻居点
        neighbor_encoded = self.input_encoder(
            neighbors.view(-1, self.input_dim)
        ).view(batch_size, self.num_neighbors, -1)  # [batch_size, num_neighbors, hidden_dim]
        
        # 拼接中心点和邻居
        all_points = torch.cat([center_encoded, neighbor_encoded], dim=1)
        
        # 局部注意力
        attn_output, _ = self.local_attention(
            center_encoded,  # query: 中心点
            all_points,      # key: 所有点
            all_points       # value: 所有点
        )
        
        # 残差连接和归一化
        h = self.norm1(center_encoded + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(h)
        h = self.norm2(h + ffn_output)
        
        # 输出
        h = h.squeeze(1)  # [batch_size, hidden_dim]
        logits = self.output_layer(h)
        
        return F.softmax(logits, dim=-1)