"""
4D Hadwiger-Nelson实验
探索4D欧几里得空间的染色数
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator import PointPairGenerator
from models.mlp_mapper import MLPColorMapper
from models.residual_net import ResidualColorNet
from losses.constraint_loss import ConstraintLoss
from training.trainer import TrainingConfig, HadwigerNelsonTrainer
from evaluation.visualizer import HadwigerNelsonVisualizer
from evaluation.validator import HadwigerNelsonValidator
from evaluation.metrics import HadwigerNelsonMetrics


def run_4d_experiment(k_colors: int = 9,
                     model_type: str = "mlp",
                     epochs: int = 2500,
                     batch_size: int = 4096,
                     hidden_dims: List[int] = [128, 256, 128],
                     output_dir: str = "./results_4d",
                     device: str = "cuda") -> Dict[str, Any]:
    """
    运行4D实验
    
    Args:
        k_colors: 颜色数量
        model_type: 模型类型 ("mlp" 或 "residual")
        epochs: 训练轮数
        batch_size: 批次大小
        hidden_dims: 隐藏层维度
        output_dir: 输出目录
        device: 设备
        
    Returns:
        results: 实验结果字典
    """
    
    print(f"\n{'='*60}")
    print(f"4D Experiment: k={k_colors}, model={model_type}")
    print('='*60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 创建数据生成器
    generator = PointPairGenerator(
        dim=4,
        batch_size=batch_size,
        space_range=(-2.5, 2.5),
        device=device,
        seed=42
    )
    
    # 创建模型
    if model_type == "mlp":
        model = MLPColorMapper(
            input_dim=4,
            num_colors=k_colors,
            hidden_dims=hidden_dims,
            activation="relu",
            use_batch_norm=True,
            dropout_rate=0.1,
            use_fourier=True,
            fourier_features=512,  # 4D需要更多特征
            fourier_sigma=5.0
        )
    elif model_type == "residual":
        model = ResidualColorNet(
            input_dim=4,
            num_colors=k_colors,
            hidden_dim=256,
            num_blocks=6,  # 4D需要更深的网络
            activation="relu",
            dropout_rate=0.1,
            use_fourier=True,
            fourier_features=512,
            fourier_sigma=5.0
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # 创建损失函数
    loss_fn = ConstraintLoss(
        conflict_weight=1.0,
        entropy_weight=0.1,
        uniformity_weight=0.01,
        spectral_weight=0.05
    )
    
    # 创建训练配置
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.001,
        weight_decay=1e-5,
        gradient_clip=1.0,
        validation_freq=100,
        save_freq=500,
        checkpoint_dir=os.path.join(output_dir, f"checkpoints_4d_k{k_colors}"),
        log_dir=os.path.join(output_dir, f"logs_4d_k{k_colors}"),
        device=device.type,
        seed=42
    )
    
    # 创建训练器并训练
    trainer = HadwigerNelsonTrainer(
        model=model,
        data_generator=generator,
        loss_fn=loss_fn,
        config=config
    )
    
    print(f"Training {model_type.upper()} model with {k_colors} colors in 4D...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    history = trainer.train()
    
    # 最终评估
    validator = HadwigerNelsonValidator(generator, device=device.type)
    final_metrics = validator.validate_model(model, num_samples=50000)
    
    # 计算所有指标
    all_metrics = HadwigerNelsonMetrics.compute_all_metrics(
        model, generator, num_samples=10000
    )
    
    # 可视化
    visualizer = HadwigerNelsonVisualizer(device=device.type)
    
    # 保存训练曲线
    fig = visualizer.visualize_training_history(
        history,
        save_path=os.path.join(output_dir, f"training_history_4d_k{k_colors}.png")
    )
    
    # 可视化4D染色方案（交互式HTML）
    fig_4d = visualizer.visualize_4d_coloring(
        model, k_colors,
        resolution=20,
        space_range=(-1.5, 1.5),
        save_path=os.path.join(output_dir, f"4d_coloring_k{k_colors}.html")
    )
    
    # 分析颜色分布
    color_analysis = validator.analyze_color_distribution(model, num_samples=10000)
    
    # 分析几何性质
    geometric_analysis = {}
    for c in range(k_colors):
        geometric_analysis[c] = validator.compute_geometric_properties(
            model, c, num_samples=5000
        )
    
    # 保存结果
    results = {
        'dimension': 4,
        'k_colors': k_colors,
        'model_type': model_type,
        'final_violation_rate': final_metrics['violation_rate'],
        'best_violation_rate': trainer.best_loss,
        'final_metrics': final_metrics,
        'all_metrics': all_metrics,
        'color_analysis': color_analysis,
        'geometric_analysis': geometric_analysis,
        'training_history': [
            {
                'epoch': h['epoch'],
                'train_loss': h['train']['total_loss'] if h.get('train') else None,
                'val_violation': h['validation']['violation_rate'] if h.get('validation') else None
            }
            for h in history
        ]
    }
    
    # 保存结果到JSON
    results_file = os.path.join(output_dir, f"results_4d_k{k_colors}.json")
    with open(results_file, 'w') as f:
        # 转换numpy数组为列表
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            else:
                return obj
        
        json.dump(convert(results), f, indent=2)
    
    print(f"\n4D Experiment completed for k={k_colors}")
    print(f"Final violation rate: {final_metrics['violation_rate']:.2f}%")
    print(f"Best violation rate: {trainer.best_loss:.2f}%")
    print(f"Results saved to: {results_file}")
    
    return results


def run_4d_color_sweep(color_range: List[int] = [5, 6, 7, 8, 9, 10, 11, 12],
                      model_type: str = "mlp",
                      epochs: int = 2000,
                      output_dir: str = "./results_4d_sweep") -> Dict[int, Dict[str, Any]]:
    """
    运行4D颜色数扫描实验
    
    Args:
        color_range: 要测试的颜色数列表
        model_type: 模型类型
        epochs: 每个实验的训练轮数
        output_dir: 输出目录
        
    Returns:
        all_results: 所有实验结果字典
    """
    
    print("=" * 80)
    print("4D Color Number Sweep Experiment")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for k in color_range:
        print(f"\nTesting k = {k}")
        print("-" * 40)
        
        try:
            results = run_4d_experiment(
                k_colors=k,
                model_type=model_type,
                epochs=epochs,
                output_dir=os.path.join(output_dir, f"k{k}"),
                device="cuda"
            )
            
            all_results[k] = results
            
        except Exception as e:
            print(f"Error testing k={k}: {e}")
            continue
    
    # 分析结果并绘制图表
    if all_results:
        analyze_4d_results(all_results, output_dir)
    
    return all_results


def analyze_4d_results(results: Dict[int, Dict[str, Any]], output_dir: str):
    """分析4D实验结果"""
    
    k_values = sorted(results.keys())
    violation_rates = [results[k]['final_violation_rate'] for k in k_values]
    best_violation_rates = [results[k]['best_violation_rate'] for k in k_values]
    
    # 绘制结果图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 冲突率 vs 颜色数
    ax1 = axes[0, 0]
    ax1.plot(k_values, violation_rates, 'bo-', linewidth=2, markersize=8, label='Final')
    ax1.plot(k_values, best_violation_rates, 'r^--', linewidth=2, markersize=8, label='Best')
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1% threshold')
    ax1.fill_between(k_values, 0, 1.0, alpha=0.1, color='green')
    ax1.set_xlabel('Number of colors (k)')
    ax1.set_ylabel('Violation Rate (%)')
    ax1.set_title('4D: Violation Rate vs Number of Colors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 使用的颜色数
    ax2 = axes[0, 1]
    used_colors = [results[k]['all_metrics']['color']['num_used_colors'] for k in k_values]
    ax2.bar(k_values, used_colors, alpha=0.7)
    ax2.plot(k_values, k_values, 'r--', label='Ideal (all colors used)')
    ax2.set_xlabel('Available colors (k)')
    ax2.set_ylabel('Actually used colors')
    ax2.set_title('Color Usage Efficiency in 4D')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 不同k值的模型确定性
    ax3 = axes[1, 0]
    
    # 计算不同k值下的最大可能熵
    max_entropies = [np.log(k) for k in k_values]
    actual_entropies = [results[k]['all_metrics']['color']['mean_entropy'] for k in k_values]
    
    ax3.plot(k_values, max_entropies, 'g--', linewidth=2, label='Max possible entropy')
    ax3.plot(k_values, actual_entropies, 'g-', linewidth=2, marker='o', label='Actual entropy')
    
    ax3.set_xlabel('Number of colors (k)')
    ax3.set_ylabel('Entropy', color='g')
    ax3.set_title('Model Uncertainty in 4D')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 总体评分
    ax4 = axes[1, 1]
    overall_scores = [results[k]['all_metrics']['overall_score'] for k in k_values]
    ax4.plot(k_values, overall_scores, 'm-', linewidth=2, marker='s', markersize=8)
    ax4.set_xlabel('Number of colors (k)')
    ax4.set_ylabel('Overall Score')
    ax4.set_title('Overall Performance in 4D')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.suptitle('4D Hadwiger-Nelson Problem Analysis', fontsize=16)
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, "4d_results_summary.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n4D results summary saved to: {plot_path}")
    
    # 计算估计的下界
    feasible_k = []
    for k in k_values:
        if results[k]['final_violation_rate'] < 1.0:  # 1%阈值
            feasible_k.append(k)
    
    if feasible_k:
        estimated_lower_bound = min(feasible_k)
        print(f"\nEstimated lower bound for χ(ℝ⁴): ≥ {estimated_lower_bound}")
        print(f"Feasible k values: {sorted(feasible_k)}")
        
        # 保存下界估计
        bound_file = os.path.join(output_dir, "estimated_lower_bound.txt")
        with open(bound_file, 'w') as f:
            f.write(f"Estimated lower bound for χ(ℝ⁴): ≥ {estimated_lower_bound}\n")
            f.write(f"Based on experiments with k = {k_values}\n")
            f.write(f"Feasible k values (violation rate < 1%): {sorted(feasible_k)}\n")
        
        print(f"Lower bound estimate saved to: {bound_file}")
    else:
        print("\nNo feasible k found in tested range")
    
    return fig


def compare_dimensions(results_2d: Dict[int, Dict[str, Any]],
                      results_3d: Dict[int, Dict[str, Any]],
                      results_4d: Dict[int, Dict[str, Any]],
                      output_dir: str = "./results_comparison"):
    """比较不同维度的结果"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 找到共同的k值
    common_k = []
    if results_2d:
        common_k = list(results_2d.keys())
    if results_3d:
        common_k = list(set(common_k) & set(results_3d.keys()))
    if results_4d:
        common_k = list(set(common_k) & set(results_4d.keys()))
    
    common_k = sorted(common_k)
    
    if len(common_k) < 2:
        print("Not enough common k values for comparison")
        return
    
    # 准备数据
    violation_rates_2d = [results_2d[k]['final_violation_rate'] for k in common_k]
    violation_rates_3d = [results_3d[k]['final_violation_rate'] for k in common_k]
    violation_rates_4d = [results_4d[k]['final_violation_rate'] for k in common_k]
    
    # 绘制比较图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 冲突率比较
    ax1 = axes[0]
    ax1.plot(common_k, violation_rates_2d, 'bo-', linewidth=2, markersize=8, label='2D')
    ax1.plot(common_k, violation_rates_3d, 'g^-', linewidth=2, markersize=8, label='3D')
    ax1.plot(common_k, violation_rates_4d, 'rs-', linewidth=2, markersize=8, label='4D')
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1% threshold')
    ax1.set_xlabel('Number of colors (k)')
    ax1.set_ylabel('Violation Rate (%)')
    ax1.set_title('Violation Rate Comparison Across Dimensions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 维度 vs 最小可行k
    ax2 = axes[1]
    dimensions = [2, 3, 4]
    min_feasible_k = []
    
    for dim, results in [(2, results_2d), (3, results_3d), (4, results_4d)]:
        if results:
            feasible_k = [k for k, r in results.items() if r['final_violation_rate'] < 1.0]
            if feasible_k:
                min_feasible_k.append(min(feasible_k))
            else:
                min_feasible_k.append(None)
        else:
            min_feasible_k.append(None)
    
    ax2.plot(dimensions, min_feasible_k, 'mo-', linewidth=2, markersize=10)
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Minimum feasible k')
    ax2.set_title('Minimum Colors Needed by Dimension')
    ax2.grid(True, alpha=0.3)
    
    # 3. 相对难度
    ax3 = axes[2]
    
    # 计算相对难度：更高维度需要的颜色数增长
    if len(min_feasible_k) >= 2:
        # 计算从2D到3D，3D到4D的增长
        growth_rates = []
        for i in range(len(min_feasible_k)-1):
            if min_feasible_k[i] and min_feasible_k[i+1]:
                growth = min_feasible_k[i+1] / min_feasible_k[i]
                growth_rates.append(growth)
        
        if growth_rates:
            x_pos = [2.5, 3.5]  # 中点位置
            ax3.bar(x_pos, growth_rates, width=0.8, alpha=0.7)
            ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Dimension transition')
            ax3.set_ylabel('Growth rate (k_{d+1} / k_d)')
            ax3.set_title('Color Requirement Growth with Dimension')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(['2D→3D', '3D→4D'])
            ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Hadwiger-Nelson Problem: Dimension Comparison', fontsize=16)
    plt.tight_layout()
    
    # 保存比较图
    plot_path = os.path.join(output_dir, "dimension_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Dimension comparison saved to: {plot_path}")
    
    # 保存比较结果
    comparison_results = {
        'common_k_values': common_k,
        'violation_rates': {
            '2d': violation_rates_2d,
            '3d': violation_rates_3d,
            '4d': violation_rates_4d
        },
        'min_feasible_k': {
            '2d': min_feasible_k[0] if len(min_feasible_k) > 0 else None,
            '3d': min_feasible_k[1] if len(min_feasible_k) > 1 else None,
            '4d': min_feasible_k[2] if len(min_feasible_k) > 2 else None
        }
    }
    
    comp_file = os.path.join(output_dir, "dimension_comparison.json")
    with open(comp_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"Comparison results saved to: {comp_file}")
    
    return fig


if __name__ == "__main__":
    # 运行4D实验
    # results = run_4d_experiment(k_colors=9, output_dir="./results_4d_single")
    
    # 运行4D颜色数扫描
    results_4d = run_4d_color_sweep(
        color_range=[5, 6, 7, 8, 9, 10, 11, 12],
        epochs=2000,
        output_dir="./results_4d_sweep"
    )
    
    # 注意：需要先运行2D和3D实验以获得比较数据
    # compare_dimensions(results_2d, results_3d, results_4d, "./results_comparison")