import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class HadwigerNelsonValidator:
    """Hadwiger-Nelson问题验证器"""
    
    def __init__(self, 
                 generator,
                 device: str = "cuda"):
        
        self.generator = generator
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    def validate_model(self, 
                      model, 
                      num_samples: int = 10000,
                      batch_size: int = 4096) -> Dict[str, Any]:
        """
        验证模型
        
        Args:
            model: 要验证的模型
            num_samples: 测试样本数
            batch_size: 批次大小
            
        Returns:
            metrics: 验证指标字典
        """
        model.eval()
        
        total_violations = 0
        total_pairs = 0
        all_distances = []
        all_color_diffs = []
        
        with torch.no_grad():
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            for _ in range(num_batches):
                # 生成点对
                p1, p2 = self.generator.get_batch()
                p1 = p1.to(self.device)
                p2 = p2.to(self.device)
                
                # 获取颜色分配
                colors1 = model.get_color_assignment(p1)
                colors2 = model.get_color_assignment(p2)
                
                # 统计冲突
                violations = (colors1 == colors2).sum().item()
                total_violations += violations
                total_pairs += len(p1)
                
                # 计算距离（应该是1）
                distances = torch.norm(p1 - p2, dim=1)
                all_distances.append(distances.cpu().numpy())
                
                # 计算颜色分布差异
                if hasattr(model, 'forward'):
                    out1 = model(p1)
                    out2 = model(p2)
                    color_diffs = torch.norm(out1 - out2, dim=1)
                    all_color_diffs.append(color_diffs.cpu().numpy())
        
        # 计算指标
        violation_rate = (total_violations / total_pairs) * 100 if total_pairs > 0 else 0
        
        if all_distances:
            all_distances = np.concatenate(all_distances)
            distance_stats = {
                'mean': float(np.mean(all_distances)),
                'std': float(np.std(all_distances)),
                'min': float(np.min(all_distances)),
                'max': float(np.max(all_distances))
            }
        else:
            distance_stats = {}
        
        if all_color_diffs:
            all_color_diffs = np.concatenate(all_color_diffs)
            color_diff_stats = {
                'mean': float(np.mean(all_color_diffs)),
                'std': float(np.std(all_color_diffs)),
                'min': float(np.min(all_color_diffs)),
                'max': float(np.max(all_color_diffs))
            }
        else:
            color_diff_stats = {}
        
        # 计算熵
        entropy = self.compute_entropy(model, num_samples=1000)
        
        return {
            'violation_rate': violation_rate,
            'total_violations': total_violations,
            'total_pairs': total_pairs,
            'distance_stats': distance_stats,
            'color_diff_stats': color_diff_stats,
            'entropy': entropy
        }
    
    def compute_entropy(self, model, num_samples: int = 1000) -> float:
        """计算模型输出的平均熵"""
        model.eval()
        
        # 随机采样点
        dim = self.generator.dim
        points = torch.randn(num_samples, dim, device=self.device)
        
        with torch.no_grad():
            outputs = model(points)
            log_outputs = torch.log(outputs + 1e-10)
            entropy = -torch.sum(outputs * log_outputs, dim=1).mean().item()
        
        return entropy
    
    def find_violating_pairs(self, 
                            model, 
                            num_samples: int = 10000,
                            max_violations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """找到违反约束的点对"""
        model.eval()
        
        violating_p1 = []
        violating_p2 = []
        
        with torch.no_grad():
            num_batches = (num_samples + self.generator.batch_size - 1) // self.generator.batch_size
            
            for _ in range(num_batches):
                if len(violating_p1) >= max_violations:
                    break
                
                p1, p2 = self.generator.get_batch()
                p1 = p1.to(self.device)
                p2 = p2.to(self.device)
                
                colors1 = model.get_color_assignment(p1)
                colors2 = model.get_color_assignment(p2)
                
                mask = (colors1 == colors2).cpu().numpy()
                
                if mask.any():
                    p1_np = p1.cpu().numpy()[mask]
                    p2_np = p2.cpu().numpy()[mask]
                    
                    violating_p1.append(p1_np)
                    violating_p2.append(p2_np)
        
        if violating_p1:
            violating_p1 = np.concatenate(violating_p1)[:max_violations]
            violating_p2 = np.concatenate(violating_p2)[:max_violations]
        else:
            violating_p1 = np.array([])
            violating_p2 = np.array([])
        
        return violating_p1, violating_p2
    
    def analyze_color_distribution(self, 
                                  model, 
                                  num_samples: int = 10000) -> Dict[str, Any]:
        """分析颜色分布"""
        model.eval()
        
        dim = self.generator.dim
        points = torch.randn(num_samples, dim, device=self.device)
        
        with torch.no_grad():
            outputs = model(points)
            color_ids = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # 统计每种颜色的点数
        num_colors = outputs.shape[1]
        color_counts = np.bincount(color_ids, minlength=num_colors)
        color_proportions = color_counts / num_samples
        
        # 计算颜色分布的均匀性
        uniform_target = np.ones(num_colors) / num_colors
        kl_divergence = np.sum(color_proportions * np.log(color_proportions / uniform_target + 1e-10))
        l2_distance = np.linalg.norm(color_proportions - uniform_target)
        
        # 计算每种颜色的空间分布
        color_centers = []
        color_spreads = []
        
        for c in range(num_colors):
            mask = (color_ids == c)
            if np.sum(mask) > 0:
                color_points = points.cpu().numpy()[mask]
                center = np.mean(color_points, axis=0)
                spread = np.std(color_points, axis=0).mean()
                
                color_centers.append(center)
                color_spreads.append(spread)
            else:
                color_centers.append(np.zeros(dim))
                color_spreads.append(0.0)
        
        return {
            'color_counts': color_counts.tolist(),
            'color_proportions': color_proportions.tolist(),
            'uniformity_kl': float(kl_divergence),
            'uniformity_l2': float(l2_distance),
            'color_centers': color_centers,
            'color_spreads': color_spreads,
            'num_used_colors': np.sum(color_counts > 0)
        }
    
    def compute_geometric_properties(self, 
                                    model, 
                                    color_id: int,
                                    num_samples: int = 5000) -> Dict[str, Any]:
        """计算特定颜色区域的几何性质"""
        model.eval()
        
        dim = self.generator.dim
        points = torch.randn(num_samples, dim, device=self.device)
        
        with torch.no_grad():
            outputs = model(points)
            color_ids = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # 提取该颜色的点
        mask = (color_ids == color_id)
        color_points = points.cpu().numpy()[mask]
        
        if len(color_points) < dim + 1:
            return {
                'num_points': len(color_points),
                'convex_hull_volume': 0.0,
                'convex_hull_area': 0.0,
                'center': np.zeros(dim).tolist(),
                'extent': np.zeros(dim).tolist()
            }
        
        # 计算凸包性质
        try:
            if dim == 2:
                hull = ConvexHull(color_points)
                area = hull.volume  # 在2D中，volume是面积
                volume = area
            elif dim == 3:
                hull = ConvexHull(color_points)
                volume = hull.volume
                area = hull.area
            else:
                # 高维，简化处理
                volume = 0.0
                area = 0.0
        except:
            volume = 0.0
            area = 0.0
        
        # 计算中心点和范围
        center = np.mean(color_points, axis=0)
        extent = np.max(color_points, axis=0) - np.min(color_points, axis=0)
        
        return {
            'num_points': len(color_points),
            'convex_hull_volume': float(volume),
            'convex_hull_area': float(area),
            'center': center.tolist(),
            'extent': extent.tolist(),
            'density': len(color_points) / (np.prod(extent) + 1e-10)
        }
    
    def visualize_violations(self, 
                            model, 
                            save_path: str = "violations.png",
                            num_samples: int = 10000,
                            max_display: int = 100):
        """可视化违反约束的点对"""
        
        if self.generator.dim != 2:
            print("可视化只支持2D")
            return
        
        # 找到违反约束的点对
        p1, p2 = self.find_violating_pairs(model, num_samples, max_display)
        
        if len(p1) == 0:
            print("没有找到违反约束的点对")
            return
        
        # 绘制
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 1. 点对连线
        ax1 = axes[0]
        for i in range(min(len(p1), max_display)):
            ax1.plot([p1[i, 0], p2[i, 0]], [p1[i, 1], p2[i, 1]], 'r-', alpha=0.5, linewidth=0.5)
            ax1.scatter(p1[i, 0], p1[i, 1], c='blue', s=10, alpha=0.7)
            ax1.scatter(p2[i, 0], p2[i, 1], c='green', s=10, alpha=0.7)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f'Violating Pairs (n={len(p1)})')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 2. 中点分布
        ax2 = axes[1]
        midpoints = (p1 + p2) / 2
        ax2.scatter(midpoints[:, 0], midpoints[:, 1], c='red', s=20, alpha=0.6)
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Midpoints of Violating Pairs')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Constraint Violations Analysis')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Violations visualization saved to {save_path}")
        
        return fig