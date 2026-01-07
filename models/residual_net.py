"""
残差网络模型
使用残差连接构建更深的网络，缓解梯度消失问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ResidualBlock(nn.Module):
    """基础的残差块"""
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 dropout_rate: float = 0.1,
                 activation: str = "relu"):
        super().__init__()
        
        # 两个线性层，带有批归一化和激活函数
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # 如果输入输出维度不同，需要投影
        self.skip_connection = None
        if in_features != out_features:
            self.skip_connection = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        identity = x
        
        # 第一个线性层
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # 第二个线性层
        out = self.linear2(out)
        out = self.bn2(out)
        
        # 跳跃连接
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)
        
        # 残差连接
        out = out + identity
        out = self.activation(out)
        
        return out


class BottleneckResidualBlock(nn.Module):
    """瓶颈残差块（减少参数）"""
    
    def __init__(self, 
                 in_features: int,
                 bottleneck_features: int,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        # 第一个线性层：降维
        self.linear1 = nn.Linear(in_features, bottleneck_features)
        self.bn1 = nn.BatchNorm1d(bottleneck_features)
        
        # 第二个线性层：瓶颈
        self.linear2 = nn.Linear(bottleneck_features, bottleneck_features)
        self.bn2 = nn.BatchNorm1d(bottleneck_features)
        
        # 第三个线性层：升维
        self.linear3 = nn.Linear(bottleneck_features, in_features)
        self.bn3 = nn.BatchNorm1d(in_features)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # 降维
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 瓶颈
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 升维
        out = self.linear3(out)
        out = self.bn3(out)
        
        # 残差连接
        out = out + identity
        out = self.relu(out)
        
        return out


class DenseResidualBlock(nn.Module):
    """密集残差块（DenseNet风格）"""
    
    def __init__(self, 
                 in_features: int,
                 growth_rate: int = 32,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        # 每个密集块有多个层
        self.linear1 = nn.Linear(in_features, growth_rate)
        self.bn1 = nn.BatchNorm1d(growth_rate)
        
        self.linear2 = nn.Linear(in_features + growth_rate, growth_rate)
        self.bn2 = nn.BatchNorm1d(growth_rate)
        
        self.linear3 = nn.Linear(in_features + 2 * growth_rate, growth_rate)
        self.bn3 = nn.BatchNorm1d(growth_rate)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        
        # 输出层
        self.output_linear = nn.Linear(in_features + 3 * growth_rate, in_features)
        self.output_bn = nn.BatchNorm1d(in_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # 第一层
        out1 = self.linear1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)
        
        # 拼接并进入第二层
        concat1 = torch.cat([x, out1], dim=1)
        out2 = self.linear2(concat1)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)
        out2 = self.dropout(out2)
        
        # 拼接并进入第三层
        concat2 = torch.cat([concat1, out2], dim=1)
        out3 = self.linear3(concat2)
        out3 = self.bn3(out3)
        out3 = self.relu(out3)
        out3 = self.dropout(out3)
        
        # 拼接所有特征
        concat3 = torch.cat([concat2, out3], dim=1)
        
        # 输出层
        out = self.output_linear(concat3)
        out = self.output_bn(out)
        
        # 残差连接
        out = out + identity
        out = self.relu(out)
        
        return out


class ResidualColorNet(nn.Module):
    """基于残差网络的颜色映射器"""
    
    def __init__(self, 
                 input_dim: int,
                 num_colors: int,
                 hidden_dims: List[int] = [128, 256, 256, 128],
                 block_type: str = "standard",
                 dropout_rate: float = 0.1,
                 activation: str = "relu",
                 use_skip_connections: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_colors = num_colors
        self.use_skip_connections = use_skip_connections
        
        # 输入层
        layers = []
        prev_dim = input_dim
        
        # 构建残差网络
        for i, hidden_dim in enumerate(hidden_dims):
            if block_type == "standard":
                block = ResidualBlock(
                    in_features=prev_dim,
                    out_features=hidden_dim,
                    dropout_rate=dropout_rate,
                    activation=activation
                )
            elif block_type == "bottleneck":
                bottleneck_dim = hidden_dim // 4
                block = BottleneckResidualBlock(
                    in_features=prev_dim,
                    bottleneck_features=bottleneck_dim,
                    dropout_rate=dropout_rate
                )
            elif block_type == "dense":
                block = DenseResidualBlock(
                    in_features=prev_dim,
                    growth_rate=32,
                    dropout_rate=dropout_rate
                )
            else:
                raise ValueError(f"Unknown block type: {block_type}")
            
            layers.append(block)
            prev_dim = hidden_dim
        
        self.residual_blocks = nn.ModuleList(layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.ReLU(inplace=True) if activation == "relu" else nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1] // 2, num_colors)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """前向传播"""
        
        # 残差块
        for block in self.residual_blocks:
            x = block(x)
        
        # 输出层
        logits = self.output_layer(x)
        
        # 温度调节
        if temperature != 1.0:
            logits = logits / temperature
        
        # Softmax
        return F.softmax(logits, dim=-1)
    
    def get_color_assignment(self, x: torch.Tensor) -> torch.Tensor:
        """获取离散颜色分配"""
        with torch.no_grad():
            probs = self.forward(x)
            return torch.argmax(probs, dim=-1)
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """获取中间特征图（用于可视化）"""
        features = []
        
        # 通过每个残差块
        for block in self.residual_blocks:
            x = block(x)
            features.append(x.clone())
        
        return features


class ResNetWithFourier(nn.Module):
    """带傅里叶特征编码的残差网络"""
    
    def __init__(self, 
                 input_dim: int,
                 num_colors: int,
                 fourier_features: int = 256,
                 fourier_sigma: float = 10.0,
                 hidden_dims: List[int] = [128, 256, 256, 128]):
        super().__init__()
        
        # 傅里叶特征编码
        from models.fourier_features import FourierFeatures
        self.fourier = FourierFeatures(
            input_dim=input_dim,
            num_features=fourier_features,
            sigma=fourier_sigma,
            include_input=True
        )
        
        # 残差网络
        fourier_output_dim = self.fourier.output_dim
        self.resnet = ResidualColorNet(
            input_dim=fourier_output_dim,
            num_colors=num_colors,
            hidden_dims=hidden_dims,
            block_type="standard",
            dropout_rate=0.1,
            activation="relu"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 傅里叶编码
        x_fourier = self.fourier(x)
        
        # 残差网络
        return self.resnet(x_fourier)


class WideResidualColorNet(nn.Module):
    """宽残差网络（宽度优先）"""
    
    def __init__(self, 
                 input_dim: int,
                 num_colors: int,
                 width: int = 512,
                 depth: int = 4,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_colors = num_colors
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 宽残差块
        self.residual_blocks = nn.ModuleList()
        for _ in range(depth):
            block = ResidualBlock(
                in_features=width,
                out_features=width,
                dropout_rate=dropout_rate,
                activation="relu"
            )
            self.residual_blocks.append(block)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.BatchNorm1d(width // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(width // 2, num_colors)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        logits = self.output_layer(x)
        return F.softmax(logits, dim=-1)