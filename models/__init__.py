"""
Hadwiger-Nelson问题模型包
"""

from .mlp_mapper import MLPColorMapper, ResidualColorMapper
from .fourier_features import FourierFeatures, LearnableFourierFeatures, PositionalEncoding
from .residual_net import ResidualColorNet
from .attention_net import (
    AttentionColorMapper,
    LocalAttentionColorMapper,
    SpatialAttention,
    MultiHeadSelfAttention
)
from .graph_net import (
    GraphColorNet,
    GeometricGraphConv,
    GeometricMessagePassing,
    DynamicGraphBuilder,
    GraphBasedColorMapper
)

__all__ = [
    # MLP相关
    'MLPColorMapper',
    'ResidualColorMapper',
    
    # 特征编码
    'FourierFeatures',
    'LearnableFourierFeatures',
    'PositionalEncoding',
    
    # 残差网络
    'ResidualColorNet',
    
    # 注意力网络
    'AttentionColorMapper',
    'LocalAttentionColorMapper',
    'SpatialAttention',
    'MultiHeadSelfAttention',
    
    # 图神经网络
    'GraphColorNet',
    'GeometricGraphConv',
    'GeometricMessagePassing',
    'DynamicGraphBuilder',
    'GraphBasedColorMapper'
]