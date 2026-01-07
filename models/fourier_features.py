"""
傅里叶特征编码
将输入坐标映射到高频特征空间，帮助网络学习高频函数
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, List


class FourierFeatures(nn.Module):
    """基本的傅里叶特征编码"""
    
    def __init__(self, 
                 input_dim: int,
                 num_features: int = 256,
                 sigma: float = 10.0,
                 include_input: bool = True,
                 learnable: bool = False):
        """
        Args:
            input_dim: 输入维度
            num_features: 傅里叶特征数量（sin和cos各一半）
            sigma: 频率矩阵的标准差
            include_input: 是否包含原始输入
            learnable: 频率矩阵是否可学习
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_features = num_features
        self.include_input = include_input
        self.learnable = learnable
        
        # 计算输出维度
        self.output_dim = num_features  # sin + cos
        if include_input:
            self.output_dim += input_dim
        
        # 初始化频率矩阵 B
        # B 的 shape: [input_dim, num_features//2]
        B = torch.randn(input_dim, num_features // 2) * sigma
        
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim] 或 [input_dim]
        返回: [batch_size, output_dim] 或 [output_dim]
        """
        # 确保是二维
        if x.dim() == 1:
            x = x.unsqueeze(0)
            unsqueeze = True
        else:
            unsqueeze = False
        
        # 计算傅里叶特征
        # proj = 2π * x @ B
        proj = 2 * math.pi * x @ self.B  # [batch_size, num_features//2]
        
        # sin和cos特征
        sin_features = torch.sin(proj)
        cos_features = torch.cos(proj)
        
        # 拼接
        fourier_features = torch.cat([sin_features, cos_features], dim=-1)
        
        # 可选：包含原始输入
        if self.include_input:
            output = torch.cat([x, fourier_features], dim=-1)
        else:
            output = fourier_features
        
        # 恢复原始维度
        if unsqueeze:
            output = output.squeeze(0)
        
        return output
    
    def get_frequencies(self) -> torch.Tensor:
        """获取频率矩阵"""
        return self.B.data if self.learnable else self.B


class GaussianFourierFeatures(FourierFeatures):
    """高斯傅里叶特征编码"""
    
    def __init__(self, 
                 input_dim: int,
                 num_features: int = 256,
                 sigma: float = 10.0,
                 include_input: bool = True,
                 learnable: bool = False):
        super().__init__(input_dim, num_features, sigma, include_input, learnable)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """与父类相同，但名称更明确"""
        return super().forward(x)


class PositionalEncoding(nn.Module):
    """位置编码（类似Transformer）"""
    
    def __init__(self, 
                 input_dim: int,
                 num_frequencies: int = 10,
                 include_input: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        
        # 预计算频率
        self.frequencies = 2.0 ** torch.arange(num_frequencies)
        
        # 计算输出维度
        self.output_dim = input_dim * (2 * num_frequencies + (1 if include_input else 0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim]
        返回: [batch_size, output_dim]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            unsqueeze = True
        else:
            unsqueeze = False
        
        batch_size = x.shape[0]
        
        # 扩展维度用于广播
        x_expanded = x.unsqueeze(-1)  # [batch_size, input_dim, 1]
        freqs = self.frequencies.to(x.device).reshape(1, 1, -1)  # [1, 1, num_freq]
        
        # 计算位置编码
        angles = x_expanded * freqs * math.pi  # [batch_size, input_dim, num_freq]
        
        sin_enc = torch.sin(angles).reshape(batch_size, -1)  # [batch_size, input_dim * num_freq]
        cos_enc = torch.cos(angles).reshape(batch_size, -1)  # [batch_size, input_dim * num_freq]
        
        # 拼接特征
        encoded = torch.cat([sin_enc, cos_enc], dim=-1)
        
        # 可选：包含原始输入
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)
        
        if unsqueeze:
            encoded = encoded.squeeze(0)
        
        return encoded


class LearnableFourierFeatures(nn.Module):
    """可学习的傅里叶特征"""
    
    def __init__(self, 
                 input_dim: int,
                 num_features: int = 256,
                 initial_sigma: float = 10.0,
                 include_input: bool = True,
                 normalize_frequencies: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_features = num_features
        self.include_input = include_input
        self.normalize_frequencies = normalize_frequencies
        
        # 输出维度
        self.output_dim = num_features
        if include_input:
            self.output_dim += input_dim
        
        # 可学习的频率矩阵
        self.B = nn.Parameter(torch.randn(input_dim, num_features // 2) * initial_sigma)
        
        # 可学习的缩放和偏移
        self.scale = nn.Parameter(torch.ones(num_features // 2))
        self.shift = nn.Parameter(torch.zeros(num_features // 2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
            unsqueeze = True
        else:
            unsqueeze = False
        
        # 归一化频率矩阵（防止梯度爆炸）
        if self.normalize_frequencies:
            B_norm = self.B / (torch.norm(self.B, dim=0, keepdim=True) + 1e-8)
        else:
            B_norm = self.B
        
        # 计算投影
        proj = 2 * math.pi * x @ B_norm  # [batch_size, num_features//2]
        
        # 应用可学习的缩放和偏移
        proj = proj * self.scale + self.shift
        
        # sin和cos特征
        sin_features = torch.sin(proj)
        cos_features = torch.cos(proj)
        
        # 拼接
        fourier_features = torch.cat([sin_features, cos_features], dim=-1)
        
        # 可选：包含原始输入
        if self.include_input:
            output = torch.cat([x, fourier_features], dim=-1)
        else:
            output = fourier_features
        
        if unsqueeze:
            output = output.squeeze(0)
        
        return output


class MultiScaleFourierFeatures(nn.Module):
    """多尺度傅里叶特征"""
    
    def __init__(self, 
                 input_dim: int,
                 num_scales: int = 4,
                 features_per_scale: int = 64,
                 sigma_range: tuple = (1.0, 100.0),
                 include_input: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_scales = num_scales
        self.include_input = include_input
        
        # 在不同尺度上生成sigma值
        sigmas = torch.linspace(sigma_range[0], sigma_range[1], num_scales)
        
        # 为每个尺度创建傅里叶特征编码器
        self.fourier_encoders = nn.ModuleList()
        for sigma in sigmas:
            encoder = FourierFeatures(
                input_dim=input_dim,
                num_features=features_per_scale,
                sigma=sigma.item(),
                include_input=False,
                learnable=False
            )
            self.fourier_encoders.append(encoder)
        
        # 计算输出维度
        self.output_dim = num_scales * features_per_scale
        if include_input:
            self.output_dim += input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
            unsqueeze = True
        else:
            unsqueeze = False
        
        # 收集所有尺度的特征
        all_features = []
        for encoder in self.fourier_encoders:
            features = encoder(x)
            all_features.append(features)
        
        # 拼接所有特征
        multi_scale_features = torch.cat(all_features, dim=-1)
        
        # 可选：包含原始输入
        if self.include_input:
            output = torch.cat([x, multi_scale_features], dim=-1)
        else:
            output = multi_scale_features
        
        if unsqueeze:
            output = output.squeeze(0)
        
        return output


class FourierMLP(nn.Module):
    """带傅里叶特征编码的MLP"""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 fourier_features: int = 256,
                 fourier_sigma: float = 10.0,
                 hidden_dims: List[int] = [128, 256, 128],
                 activation: str = "relu"):
        super().__init__()
        
        # 傅里叶特征编码
        self.fourier = FourierFeatures(
            input_dim=input_dim,
            num_features=fourier_features,
            sigma=fourier_sigma,
            include_input=True,
            learnable=False
        )
        
        # MLP
        layers = []
        prev_dim = self.fourier.output_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation == "selu":
                layers.append(nn.SELU(inplace=True))
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 傅里叶编码
        x_fourier = self.fourier(x)
        
        # MLP
        return self.mlp(x_fourier)


def test_fourier_features():
    """测试傅里叶特征编码"""
    import matplotlib.pyplot as plt
    
    # 创建测试数据
    x = torch.linspace(-2, 2, 1000).unsqueeze(1)  # [1000, 1]
    
    # 测试不同的傅里叶编码器
    encoders = {
        "Standard Fourier": FourierFeatures(1, 64, sigma=10.0),
        "Positional Encoding": PositionalEncoding(1, 8),
        "Learnable Fourier": LearnableFourierFeatures(1, 64),
        "Multi-scale Fourier": MultiScaleFourierFeatures(1, 4, 16)
    }
    
    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, (name, encoder) in enumerate(encoders.items()):
        ax = axes[idx]
        
        # 编码
        encoded = encoder(x)
        
        # 绘制前几个特征
        num_features_to_plot = min(5, encoded.shape[1])
        for i in range(num_features_to_plot):
            ax.plot(x.numpy(), encoded[:, i].detach().numpy(), 
                   label=f'Feature {i+1}', alpha=0.7)
        
        ax.set_xlabel('Input x')
        ax.set_ylabel('Feature value')
        ax.set_title(f'{name} (dim={encoded.shape[1]})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # 运行测试
    fig = test_fourier_features()
    fig.savefig("fourier_features_test.png")
    print("Fourier features test completed. Plot saved to fourier_features_test.png")