# Hadwiger-Nelson ML Solver

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> 使用深度学习探索欧几里得空间的染色数（The Chromatic Number of the Plane）。

## 问题背景

**Hadwiger-Nelson 问题**：在平面上，如果任意两个距离为 1 的点必须染上不同的颜色，最少需要几种颜色？

目前数学界的已知边界为：
$$4 \le \chi(\mathbb{R}^2) \le 7$$
*(注：De Grey 在 2018 年证明了下界至少为 5，但该项目旨在验证 ML 是否能自动发现这些反例结构)*

本项目通过构建神经网络 $f: \mathbb{R}^n \rightarrow \Delta^{k-1}$，将离散的染色问题转化为连续优化问题，尝试在高维空间（2D, 3D, 4D）寻找无冲突的染色方案，从而逼近色数的下界。

## 核心特性

本项目不仅仅是一个简单的 MLP，还集成了针对几何问题的深度优化策略：

* **多种模型架构**：
    * **MLP Mapper**: 基础坐标映射网络。
    * **Residual Net**: 深层残差网络，处理复杂边界。
    * **Graph GNN**: 基于图神经网络的拓扑结构学习。
* **傅里叶特征映射 (Fourier Features)**：
    * 利用 Gaussian Fourier Mapping 解决神经网络的 "Spectral Bias" 问题，使其能学习高频的颜色突变边界。
* **难例挖掘 (Hard Example Mining)**：
    * **主动学习**：在训练过程中自动识别高冲突区域。
    * **几何硬例**：内置 Moser Spindle、Golomb Graph 等经典数学反例结构，用于对抗训练。
* **复合损失函数 (Constraint Loss)**：
    * $L_{total} = \lambda_1 L_{conflict} + \lambda_2 L_{entropy} + \lambda_3 L_{smoothness}$
    * 结合了冲突惩罚、熵正则化（迫使输出确定性颜色）和谱平滑损失。

## 方法细节

我们定义神经网络 $f_\theta(x)$ 输出点 $x$ 属于 $k$ 种颜色的概率分布 $P$。损失函数设计如下：

1. 冲突损失 (Conflict Loss)：对于距离 $\|x_i - x_j\| = 1$ 的点对，惩罚其颜色分布的相似度（点积）。

$$ L_{conflict} = \frac{1}{N} \sum_{(i,j)} \langle f_\theta(x_i), f_\theta(x_j) \rangle $$

2. 熵损失 (Entropy Loss)：为了得到“硬染色”（非黑即白），我们最小化预测分布的熵。

$$ L_{entropy} = - \frac{1}{N} \sum_i \sum_c p_{i,c} \log p_{i,c} $$

结合 温度退火 (Temperature Annealing) 策略，在训练初期允许模糊边界，后期强迫模型做出离散决策。

## 实验结果 (Results)

### 2D 平面染色 (k=7)
*(在此处插入 `results/2d_coloring_k7.png` 的图片)*
> 模型成功找到了 2D 平面的 7 色平铺方案，形成了类似六边形的蜂窝结构。

### 3D/4D 空间探索
| 维度 | 测试颜色数 (k) | 是否可行 (Violation < 1%) | 估计下界 |
| :--- | :--- | :--- | :--- |
| 2D | 3, 4, 5, 6, 7 | k=4 可行 | $\chi \ge 4$ |
| 3D | 5, 6, 7, ..., 10 | ... | ... |
| 4D | ... | ... | ... |
## 快速开始 (Quick Start)

### 1. 环境安装

建议使用 Conda 创建虚拟环境：

```bash
conda create -n graph_color python=3.8
conda activate graph_color
pip install -r requirements.txt
```

### 2. 更多参考

```bash
# 1. 基础运行 (使用默认 config.yaml 配置)
# 默认运行 2D, 3D, 4D 实验
python main.py

# 2. 指定维度和颜色数
# 仅运行 2D 实验，强制测试 3, 4, 5, 6 种颜色
python main.py --dims 2 --colors 3 4 5 6

# 3. 快速测试 (通过减少 Epochs 实现)
# 验证代码是否跑得通，而不是为了得到结果
python main.py --dims 2 --epochs 10 --batch-size 1024

# 4. 指定自定义配置文件
# 这是修改模型、Loss、傅里叶参数的正确方式！
# 你需要先复制一份 config.yaml 改名为 config_residual.yaml 等
python main.py --config config/config_custom.yaml

# 5. 运行单维度独立脚本
python experiments/experiment_2d.py
python experiments/experiment_3d.py
python experiments/experiment_4d.py
```

## 项目结构：

```text
hadwiger_nelson_ml/
├── config/               # 实验参数配置
├── data/                 # 数据流管线
│   ├── generator.py         # 核心生成器 (Unit Distance Pairs)
│   └── hard_examples.py     # 难例挖掘与对抗样本 (Active Learning)
├── models/               # 神经网络架构
│   ├── mlp_mapper.py        # 基础 MLP 映射器
│   ├── fourier_features.py  # 傅里叶特征编码 (解决 Spectral Bias)
│   ├── graph_net.py         # 图神经网络 (GNN)
│   └── ...                  # ResNet, Attention 等变体
├── losses/               # 损失函数定义
│   ├── constraint_loss.py   # 染色冲突与熵损失 (核心 Loss)
│   ├── spectral_loss.py     # 谱平滑正则化
│   └── topological_loss.py  # 拓扑持续性损失
├── training/             # 训练循环与优化
│   ├── trainer.py           # 训练器 (Trainer)
│   └── ...                  # Optimizer, Scheduler
├── evaluation/           # 验证与可视化
│   ├── visualizer.py        # 2D/3D/4D 染色效果绘图
│   └── validator.py         # 冲突率验证器
├── experiments/          # 自动化实验脚本 (2D/3D/4D)
├── main.py               # 项目主入口
├── requirements.txt      # 依赖环境
└── utils等
```

## License

MIT License
