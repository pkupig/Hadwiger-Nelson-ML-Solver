import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, homogeneity_score, completeness_score
from scipy.spatial.distance import cdist


class HadwigerNelsonMetrics:
    """Hadwiger-Nelson问题评估指标"""
    
    @staticmethod
    def compute_conflict_metrics(model, generator, num_samples: int = 10000) -> Dict[str, float]:
        """计算冲突相关的指标"""
        model.eval()
        
        total_violations = 0
        total_pairs = 0
        conflict_distances = []
        
        with torch.no_grad():
            num_batches = (num_samples + generator.batch_size - 1) // generator.batch_size
            
            for _ in range(num_batches):
                p1, p2 = generator.get_batch()
                p1 = p1.to(next(model.parameters()).device)
                p2 = p2.to(next(model.parameters()).device)
                
                colors1 = model.get_color_assignment(p1)
                colors2 = model.get_color_assignment(p2)
                
                mask = (colors1 == colors2)
                violations = mask.sum().item()
                total_violations += violations
                total_pairs += len(p1)
                
                if violations > 0:
                    # 计算冲突点对的距离（应该是1）
                    conflict_p1 = p1[mask]
                    conflict_p2 = p2[mask]
                    distances = torch.norm(conflict_p1 - conflict_p2, dim=1)
                    conflict_distances.extend(distances.cpu().numpy().tolist())
        
        # 计算指标
        metrics = {
            'violation_rate': (total_violations / total_pairs) * 100 if total_pairs > 0 else 0,
            'total_violations': total_violations,
            'total_pairs': total_pairs
        }
        
        if conflict_distances:
            metrics.update({
                'conflict_distance_mean': float(np.mean(conflict_distances)),
                'conflict_distance_std': float(np.std(conflict_distances)),
                'conflict_distance_min': float(np.min(conflict_distances)),
                'conflict_distance_max': float(np.max(conflict_distances))
            })
        
        return metrics
    
    @staticmethod
    def compute_color_metrics(model, points: torch.Tensor) -> Dict[str, Any]:
        """计算颜色相关的指标"""
        model.eval()
        
        with torch.no_grad():
            outputs = model(points)
            color_ids = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # 计算熵
            log_outputs = torch.log(outputs + 1e-10)
            entropy = -torch.sum(outputs * log_outputs, dim=1)
            mean_entropy = entropy.mean().item()
            entropy_std = entropy.std().item()
            
            # 计算置信度
            confidence = torch.max(outputs, dim=1)[0]
            mean_confidence = confidence.mean().item()
            confidence_std = confidence.std().item()
        
        # 颜色分布
        num_colors = outputs.shape[1]
        color_counts = np.bincount(color_ids, minlength=num_colors)
        color_proportions = color_counts / len(points)
        
        # 均匀性指标
        uniform_target = np.ones(num_colors) / num_colors
        kl_divergence = np.sum(color_proportions * np.log(color_proportions / uniform_target + 1e-10))
        l2_distance = np.linalg.norm(color_proportions - uniform_target)
        
        # 使用的颜色数
        num_used_colors = np.sum(color_counts > 0)
        
        return {
            'mean_entropy': mean_entropy,
            'entropy_std': entropy_std,
            'mean_confidence': mean_confidence,
            'confidence_std': confidence_std,
            'color_distribution': color_proportions.tolist(),
            'num_used_colors': int(num_used_colors),
            'color_uniformity_kl': float(kl_divergence),
            'color_uniformity_l2': float(l2_distance)
        }
    
    @staticmethod
    def compute_spatial_metrics(model, 
                               points: torch.Tensor,
                               color_ids: np.ndarray) -> Dict[str, Any]:
        """计算空间分布指标"""
        
        points_np = points.cpu().numpy()
        unique_colors = np.unique(color_ids)
        num_colors = len(unique_colors)
        
        if num_colors == 0:
            return {}
        
        # 计算每个颜色的空间中心
        color_centers = []
        color_covariances = []
        color_extents = []
        
        for color in unique_colors:
            mask = (color_ids == color)
            if np.sum(mask) > 0:
                color_points = points_np[mask]
                center = np.mean(color_points, axis=0)
                cov = np.cov(color_points.T) if len(color_points) > 1 else np.eye(points_np.shape[1])
                extent = np.max(color_points, axis=0) - np.min(color_points, axis=0)
                
                color_centers.append(center)
                color_covariances.append(cov)
                color_extents.append(extent)
            else:
                color_centers.append(np.zeros(points_np.shape[1]))
                color_covariances.append(np.eye(points_np.shape[1]))
                color_extents.append(np.zeros(points_np.shape[1]))
        
        # 计算颜色中心之间的距离
        if len(color_centers) > 1:
            center_distances = cdist(color_centers, color_centers)
            np.fill_diagonal(center_distances, np.inf)  # 去掉对角线
            min_center_distance = np.min(center_distances)
            mean_center_distance = np.mean(center_distances[center_distances < np.inf])
        else:
            min_center_distance = 0.0
            mean_center_distance = 0.0
        
        # 计算空间分离度
        separation_scores = []
        for i in range(num_colors):
            for j in range(i+1, num_colors):
                # 简化版分离度计算
                dist = np.linalg.norm(color_centers[i] - color_centers[j])
                separation_scores.append(dist)
        
        mean_separation = np.mean(separation_scores) if separation_scores else 0.0
        
        return {
            'num_colors': int(num_colors),
            'min_center_distance': float(min_center_distance),
            'mean_center_distance': float(mean_center_distance),
            'mean_separation': float(mean_separation),
            'color_centers': [c.tolist() for c in color_centers],
            'color_extents': [e.tolist() for e in color_extents]
        }
    
    @staticmethod
    def compute_graph_metrics(model, 
                            generator,
                            num_samples: int = 5000) -> Dict[str, Any]:
        """计算图相关指标"""
        
        # 生成点集
        dim = generator.dim
        points = torch.randn(num_samples, dim, device=next(model.parameters()).device)
        
        model.eval()
        with torch.no_grad():
            color_ids = model.get_color_assignment(points).cpu().numpy()
        
        points_np = points.cpu().numpy()
        
        # 构建颜色邻接图
        # 对于每个点，找到最近的k个不同颜色的点
        k = min(10, num_samples // 10)
        
        # 计算距离矩阵
        dist_matrix = cdist(points_np, points_np)
        
        # 对不同颜色的点对进行分析
        same_color_edges = []
        diff_color_edges = []
        
        for i in range(num_samples):
            # 找到最近的k个点
            nearest_indices = np.argsort(dist_matrix[i])[1:k+1]  # 去掉自己
            
            for j in nearest_indices:
                if color_ids[i] == color_ids[j]:
                    same_color_edges.append(dist_matrix[i, j])
                else:
                    diff_color_edges.append(dist_matrix[i, j])
        
        metrics = {
            'num_points': num_samples,
            'num_same_color_edges': len(same_color_edges),
            'num_diff_color_edges': len(diff_color_edges)
        }
        
        if same_color_edges:
            metrics.update({
                'same_color_distance_mean': float(np.mean(same_color_edges)),
                'same_color_distance_std': float(np.std(same_color_edges)),
                'same_color_distance_min': float(np.min(same_color_edges)),
                'same_color_distance_max': float(np.max(same_color_edges))
            })
        
        if diff_color_edges:
            metrics.update({
                'diff_color_distance_mean': float(np.mean(diff_color_edges)),
                'diff_color_distance_std': float(np.std(diff_color_edges)),
                'diff_color_distance_min': float(np.min(diff_color_edges)),
                'diff_color_distance_max': float(np.max(diff_color_edges))
            })
        
        return metrics
    
    @staticmethod
    def compute_all_metrics(model, 
                           generator,
                           num_samples: int = 10000) -> Dict[str, Any]:
        """计算所有指标"""
        
        # 生成测试点
        dim = generator.dim
        points = torch.randn(num_samples, dim, device=next(model.parameters()).device)
        
        # 冲突指标
        conflict_metrics = HadwigerNelsonMetrics.compute_conflict_metrics(
            model, generator, num_samples
        )
        
        # 颜色指标
        color_metrics = HadwigerNelsonMetrics.compute_color_metrics(model, points)
        
        # 空间指标
        with torch.no_grad():
            color_ids = model.get_color_assignment(points).cpu().numpy()
        spatial_metrics = HadwigerNelsonMetrics.compute_spatial_metrics(
            model, points, color_ids
        )
        
        # 图指标
        graph_metrics = HadwigerNelsonMetrics.compute_graph_metrics(
            model, generator, num_samples // 2
        )
        
        # 合并所有指标
        all_metrics = {}
        all_metrics.update({'conflict': conflict_metrics})
        all_metrics.update({'color': color_metrics})
        all_metrics.update({'spatial': spatial_metrics})
        all_metrics.update({'graph': graph_metrics})
        
        # 计算总体评分
        overall_score = HadwigerNelsonMetrics._compute_overall_score(all_metrics)
        all_metrics['overall_score'] = overall_score
        
        return all_metrics
    
    @staticmethod
    def _compute_overall_score(metrics: Dict[str, Any]) -> float:
        """计算总体评分"""
        
        score = 0.0
        weights = {
            'violation_rate': -1.0,  # 越低越好
            'mean_entropy': -0.5,    # 越低越好（确定性高）
            'mean_confidence': 0.5,  # 越高越好
            'num_used_colors': 0.1,  # 使用的颜色数（适中）
            'color_uniformity_l2': -0.3,  # 越均匀越好
            'mean_separation': 0.2   # 颜色分离度越高越好
        }
        
        # 冲突率
        if 'conflict' in metrics and 'violation_rate' in metrics['conflict']:
            vr = metrics['conflict']['violation_rate']
            score += weights['violation_rate'] * min(vr, 100) / 100
        
        # 颜色指标
        if 'color' in metrics:
            color_metrics = metrics['color']
            
            if 'mean_entropy' in color_metrics:
                entropy = color_metrics['mean_entropy']
                # 熵在[0, log(num_colors)]之间
                max_entropy = np.log(color_metrics.get('num_used_colors', 7))
                if max_entropy > 0:
                    score += weights['mean_entropy'] * (entropy / max_entropy)
            
            if 'mean_confidence' in color_metrics:
                confidence = color_metrics['mean_confidence']
                score += weights['mean_confidence'] * confidence
            
            if 'num_used_colors' in color_metrics:
                num_colors = color_metrics['num_used_colors']
                # 理想情况是使用所有颜色
                target_colors = 7  # 假设目标颜色数为7
                color_score = 1 - abs(num_colors - target_colors) / target_colors
                score += weights['num_used_colors'] * color_score
            
            if 'color_uniformity_l2' in color_metrics:
                uniformity = color_metrics['color_uniformity_l2']
                # L2距离在[0, sqrt(2)]之间
                score += weights['color_uniformity_l2'] * (uniformity / np.sqrt(2))
        
        # 空间指标
        if 'spatial' in metrics and 'mean_separation' in metrics['spatial']:
            separation = metrics['spatial']['mean_separation']
            # 归一化分离度（假设最大为5）
            score += weights['mean_separation'] * min(separation / 5, 1.0)
        
        # 归一化到[0, 1]
        score = (score + 2) / 4  # 假设分数在[-2, 2]之间
        
        return max(0.0, min(1.0, score))