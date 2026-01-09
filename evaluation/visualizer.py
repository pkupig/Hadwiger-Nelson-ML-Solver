import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class HadwigerNelsonVisualizer:
    """Hadwiger-Nelson问题可视化工具"""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    def visualize_2d_coloring(self, 
                            model, 
                            k_colors: int,
                            resolution: int = 300,
                            space_range: Tuple[float, float] = (-3, 3),
                            save_path: str = "2d_coloring.png") -> plt.Figure:
        """
        可视化2D染色方案
        
        Args:
            model: 训练好的模型
            k_colors: 颜色数量
            resolution: 网格分辨率
            space_range: 空间范围
            save_path: 保存路径
            
        Returns:
            fig: matplotlib图形
        """
        model.eval()
        
        # 生成网格
        x = np.linspace(space_range[0], space_range[1], resolution)
        y = np.linspace(space_range[0], space_range[1], resolution)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # 预测颜色
        with torch.no_grad():
            points_tensor = torch.FloatTensor(grid_points).to(self.device)
            preds = model(points_tensor)
            color_ids = torch.argmax(preds, dim=1).cpu().numpy()
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. 染色方案
        ax1 = axes[0]
        scatter = ax1.scatter(xx.ravel(), yy.ravel(), 
                            c=color_ids, 
                            cmap='tab20',
                            s=1, 
                            alpha=0.8)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f'2D Coloring Plan (k={k_colors})')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Color ID')
        
        # 2. 不确定性（熵）图
        ax2 = axes[1]
        with torch.no_grad():
            entropy = -torch.sum(preds * torch.log(preds + 1e-10), dim=1).cpu().numpy()
        
        entropy_img = entropy.reshape(xx.shape)
        im = ax2.imshow(entropy_img, 
                       extent=[space_range[0], space_range[1], space_range[0], space_range[1]],
                       origin='lower', 
                       cmap='viridis',
                       aspect='auto')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Uncertainty (Entropy) Map')
        plt.colorbar(im, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Hadwiger-Nelson 2D Coloring with k={k_colors}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"2D可视化已保存到 {save_path}")
        
        return fig
    
    def visualize_3d_coloring(self, 
                            model, 
                            k_colors: int,
                            resolution: int = 50,
                            space_range: Tuple[float, float] = (-2, 2),
                            save_path: str = "3d_coloring.png",
                            projection_slice: Optional[int] = None) -> plt.Figure:
        """
        可视化3D染色方案
        
        Args:
            model: 训练好的模型
            k_colors: 颜色数量
            resolution: 网格分辨率（降低以节省内存）
            space_range: 空间范围
            save_path: 保存路径
            projection_slice: 如果提供，显示特定Z切片的投影
            
        Returns:
            fig: matplotlib图形
        """
        model.eval()
        
        if projection_slice is not None:
            # 显示特定Z切片的2D投影
            fig = self._visualize_3d_slice(model, k_colors, resolution, space_range, 
                                         slice_dim=2, slice_value=projection_slice)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            return fig
        
        # 生成3D网格（降低分辨率以节省内存）
        x = np.linspace(space_range[0], space_range[1], resolution)
        y = np.linspace(space_range[0], space_range[1], resolution)
        z = np.linspace(space_range[0], space_range[1], resolution)
        
        # 由于内存限制，我们采样点而不是创建完整网格
        num_samples = 10000
        samples = np.random.uniform(space_range[0], space_range[1], (num_samples, 3))
        
        # 预测颜色
        with torch.no_grad():
            points_tensor = torch.FloatTensor(samples).to(self.device)
            preds = model(points_tensor)
            color_ids = torch.argmax(preds, dim=1).cpu().numpy()
        
        # 创建3D图形
        fig = plt.figure(figsize=(14, 10))
        
        # 1. 3D散点图
        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
                              c=color_ids, cmap='tab20', s=10, alpha=0.6)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'3D Coloring (k={k_colors})')
        
        # 2. XY平面投影
        ax2 = fig.add_subplot(222)
        scatter2 = ax2.scatter(samples[:, 0], samples[:, 1], c=color_ids, 
                              cmap='tab20', s=10, alpha=0.6)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('XY Plane Projection')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # 3. XZ平面投影
        ax3 = fig.add_subplot(224)
        scatter3 = ax3.scatter(samples[:, 0], samples[:, 2], c=color_ids,
                              cmap='tab20', s=10, alpha=0.6)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title('XZ Plane Projection')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f'Hadwiger-Nelson 3D Coloring with k={k_colors}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D可视化已保存到 {save_path}")
        
        return fig
    
    def _visualize_3d_slice(self, 
                          model, 
                          k_colors: int,
                          resolution: int,
                          space_range: Tuple[float, float],
                          slice_dim: int = 2,  # 0:X, 1:Y, 2:Z
                          slice_value: float = 0.0) -> plt.Figure:
        """可视化3D空间的2D切片"""
        # 创建2D网格
        dims = [0, 1, 2]
        dims.remove(slice_dim)
        
        x = np.linspace(space_range[0], space_range[1], resolution)
        y = np.linspace(space_range[0], space_range[1], resolution)
        xx, yy = np.meshgrid(x, y)
        
        # 创建3D点（固定切片维度的值）
        points = np.zeros((resolution * resolution, 3))
        points[:, dims[0]] = xx.ravel()
        points[:, dims[1]] = yy.ravel()
        points[:, slice_dim] = slice_value
        
        # 预测颜色
        with torch.no_grad():
            points_tensor = torch.FloatTensor(points).to(self.device)
            preds = model(points_tensor)
            color_ids = torch.argmax(preds, dim=1).cpu().numpy()
        
        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 染色方案
        ax1 = axes[0]
        scatter1 = ax1.scatter(xx.ravel(), yy.ravel(), c=color_ids, 
                              cmap='tab20', s=1, alpha=0.8)
        ax1.set_xlabel(['X', 'Y', 'Z'][dims[0]])
        ax1.set_ylabel(['X', 'Y', 'Z'][dims[1]])
        ax1.set_title(f'Slice at {["X", "Y", "Z"][slice_dim]}={slice_value:.2f}')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        plt.colorbar(scatter1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
        
        # 不确定性图
        ax2 = axes[1]
        entropy = -torch.sum(preds * torch.log(preds + 1e-10), dim=1).cpu().numpy()
        entropy_img = entropy.reshape(xx.shape)
        
        im = ax2.imshow(entropy_img, 
                       extent=[space_range[0], space_range[1], space_range[0], space_range[1]],
                       origin='lower', 
                       cmap='viridis',
                       aspect='auto')
        ax2.set_xlabel(['X', 'Y', 'Z'][dims[0]])
        ax2.set_ylabel(['X', 'Y', 'Z'][dims[1]])
        ax2.set_title('Uncertainty (Entropy)')
        plt.colorbar(im, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
        
        plt.suptitle(f'3D Coloring Slice (k={k_colors})')
        plt.tight_layout()
        
        return fig
    
    def visualize_4d_coloring(self, 
                            model, 
                            k_colors: int,
                            resolution: int = 20,
                            space_range: Tuple[float, float] = (-1.5, 1.5),
                            save_path: str = "4d_coloring.html") -> go.Figure:
        """
        可视化4D染色方案（使用交互式3D+颜色）
        
        Args:
            model: 训练好的模型
            k_colors: 颜色数量
            resolution: 网格分辨率
            space_range: 空间范围
            save_path: 保存路径（HTML文件）
            
        Returns:
            fig: plotly图形
        """
        model.eval()
        
        # 由于4D可视化困难，我们采样点并用颜色表示第4维
        # 方法：固定第4维的值，显示3D投影
        
        num_samples = 5000
        samples = np.random.uniform(space_range[0], space_range[1], (num_samples, 4))
        
        # 预测颜色
        with torch.no_grad():
            points_tensor = torch.FloatTensor(samples).to(self.device)
            preds = model(points_tensor)
            color_ids = torch.argmax(preds, dim=1).cpu().numpy()
        
        # 创建plotly图形
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('3D Projection (W as Color)', 
                           'XY Plane (Z as Color)',
                           'XZ Plane (W as Color)',
                           'YW Plane (Z as Color)'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
                  [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # 1. 3D散点图：XYZ空间，W维度用颜色表示
        fig.add_trace(
            go.Scatter3d(
                x=samples[:, 0], y=samples[:, 1], z=samples[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=samples[:, 3],  # W维度作为颜色
                    colorscale='Viridis',
                    opacity=0.6,
                    showscale=True,
                    colorbar=dict(title='W value')
                ),
                text=[f'Color: {c}<br>W: {w:.2f}' for c, w in zip(color_ids, samples[:, 3])],
                hoverinfo='text'
            ),
            row=1, col=1
        )
        
        # 2. XY平面：Z维度用颜色表示
        fig.add_trace(
            go.Scatter(
                x=samples[:, 0], y=samples[:, 1],
                mode='markers',
                marker=dict(
                    size=5,
                    color=samples[:, 2],  # Z维度作为颜色
                    colorscale='Plasma',
                    opacity=0.6,
                    showscale=True,
                    colorbar=dict(title='Z value')
                ),
                text=[f'Color: {c}<br>Z: {z:.2f}' for c, z in zip(color_ids, samples[:, 2])],
                hoverinfo='text'
            ),
            row=1, col=2
        )
        
        # 3. XZ平面：W维度用颜色表示
        fig.add_trace(
            go.Scatter(
                x=samples[:, 0], y=samples[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=samples[:, 3],  # W维度作为颜色
                    colorscale='Rainbow',
                    opacity=0.6,
                    showscale=True,
                    colorbar=dict(title='W value')
                ),
                text=[f'Color: {c}<br>W: {w:.2f}' for c, w in zip(color_ids, samples[:, 3])],
                hoverinfo='text'
            ),
            row=2, col=1
        )
        
        # 4. YW平面：Z维度用颜色表示
        fig.add_trace(
            go.Scatter(
                x=samples[:, 1], y=samples[:, 3],
                mode='markers',
                marker=dict(
                    size=5,
                    color=samples[:, 2],  # Z维度作为颜色
                    colorscale='Hot',
                    opacity=0.6,
                    showscale=True,
                    colorbar=dict(title='Z value')
                ),
                text=[f'Color: {c}<br>Z: {z:.2f}' for c, z in zip(color_ids, samples[:, 2])],
                hoverinfo='text'
            ),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title=f'4D Hadwiger-Nelson Coloring (k={k_colors})',
            showlegend=False,
            height=800
        )
        
        # 更新轴标签
        fig.update_scenes(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', row=1, col=1)
        fig.update_xaxes(title_text='X', row=1, col=2)
        fig.update_yaxes(title_text='Y', row=1, col=2)
        fig.update_xaxes(title_text='X', row=2, col=1)
        fig.update_yaxes(title_text='Z', row=2, col=1)
        fig.update_xaxes(title_text='Y', row=2, col=2)
        fig.update_yaxes(title_text='W', row=2, col=2)
        
        # 保存为HTML
        fig.write_html(save_path)
        print(f"4D交互式可视化已保存到 {save_path}")
        
        return fig
    
    def visualize_training_history(self, 
                                 history: List[dict],
                                 save_path: str = "training_history.png") -> plt.Figure:
        """可视化训练历史"""
        epochs = [h['epoch'] for h in history if h.get('train')]
        train_losses = [h['train']['total_loss'] for h in history if h.get('train')]
        conflict_losses = [h['train']['conflict_loss'] for h in history if h.get('train')]
        
        val_epochs = [h['epoch'] for h in history if h.get('validation')]
        val_violations = [h['validation']['violation_rate'] for h in history if h.get('validation')]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 训练损失
        ax1 = axes[0, 0]
        ax1.plot(epochs, train_losses, 'b-', label='Total Loss', linewidth=2)
        ax1.plot(epochs, conflict_losses, 'r--', label='Conflict Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. 验证冲突率
        ax2 = axes[0, 1]
        if val_epochs:
            ax2.plot(val_epochs, val_violations, 'g-', linewidth=2, marker='o')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Violation Rate (%)')
            ax2.set_title('Validation Violation Rate')
            ax2.grid(True, alpha=0.3)
        
        # 3. 学习率
        ax3 = axes[1, 0]
        lrs = [h['train']['learning_rate'] for h in history if h.get('train')]
        ax3.plot(epochs, lrs, 'm-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. 损失分布直方图
        ax4 = axes[1, 1]
        ax4.hist(train_losses, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax4.set_xlabel('Loss Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Training Loss')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Training History Analysis')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练历史图已保存到 {save_path}")
        
        return fig


class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, results: dict):
        self.results = results
    
    def generate_summary_table(self) -> str:
        """生成结果摘要表格"""
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("Hadwiger-Nelson Problem Results Summary")
        summary_lines.append("=" * 80)
        
        for dim in sorted(self.results.keys()):
            summary_lines.append(f"\nDimension {dim}D:")
            summary_lines.append("-" * 40)
            summary_lines.append(f"{'Colors (k)':<10} {'Violation Rate (%)':<20} {'Status':<15}")
            summary_lines.append("-" * 40)
            
            for k in sorted(self.results[dim].keys()):
                violation_rate = self.results[dim][k]['final_violation_rate']
                status = "FEASIBLE" if violation_rate < 1.0 else "INFEASIBLE"
                summary_lines.append(f"{k:<10} {violation_rate:<20.2f} {status:<15}")
        
        summary_lines.append("\n" + "=" * 80)
        summary_lines.append("Estimated Lower Bounds:")
        summary_lines.append("-" * 40)
        
        for dim in sorted(self.results.keys()):
            # ---------------------------------------------------------
            # Feasible (Violation ≈ 0) => Upper Bound (我们构造了解)
            # Infeasible (Violation > 0) => Lower Bound Hint (我们找不到解)
            # ---------------------------------------------------------
            
            dim_res = self.results[dim]
            feasible_k = [k for k in dim_res if dim_res[k]['final_violation_rate'] < 1e-3] # 收紧阈值
            infeasible_k = [k for k in dim_res if dim_res[k]['final_violation_rate'] >= 1e-3]
            
            if feasible_k:
                upper = min(feasible_k)
                summary_lines.append(f"{dim}D: Upper Bound ≤ {upper} (Constructed)")
            else:       
                summary_lines.append(f"{dim}D: No Upper Bound found in range")
                
            if infeasible_k:
                # 如果 k 是不可行的，且我们尽力了，那么 χ 可能 > k
                lower_hint = max(infeasible_k) + 1
                summary_lines.append(f"{dim}D: Lower Bound Hint ≥ {lower_hint} (Based on failure at k={max(infeasible_k)})")
        
        return "\n".join(summary_lines)
    
    def plot_violation_vs_k(self, save_path: str = "violation_vs_k.png") -> plt.Figure:
        """绘制冲突率 vs k的曲线"""
        fig, axes = plt.subplots(1, len(self.results), figsize=(5*len(self.results), 5))
        
        if len(self.results) == 1:
            axes = [axes]
        
        for idx, (dim, dim_results) in enumerate(sorted(self.results.items())):
            ax = axes[idx]
            
            k_values = sorted(dim_results.keys())
            violation_rates = [dim_results[k]['final_violation_rate'] for k in k_values]
            
            ax.plot(k_values, violation_rates, 'bo-', linewidth=2, markersize=8)
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1% threshold')
            ax.fill_between(k_values, 0, 1.0, alpha=0.1, color='green', label='Feasible region')
            
            ax.set_xlabel('Number of colors (k)')
            ax.set_ylabel('Violation Rate (%)')
            ax.set_title(f'{dim}D Space')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 标记已知结果（2D）
            if dim == 2:
                ax.axvline(x=4, color='orange', linestyle=':', alpha=0.5, label='Known: χ≥4')
                ax.axvline(x=7, color='green', linestyle=':', alpha=0.5, label='Known: χ≤7')
        
        plt.suptitle('Violation Rate vs Number of Colors')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig