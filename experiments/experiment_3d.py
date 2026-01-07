"""
3D Hadwiger-Nelson实验
探索3D欧几里得空间的染色数
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


def run_3d_experiment(k_colors: int = 7,
                     model_type: str = "mlp",
                     epochs: int = 2000,
                     batch_size: int = 4096,
                     hidden_dims: List[int] = [128, 256, 128],
                     output_dir: str = "./results_3d",
                     device: str = "cuda") -> Dict[str, Any]:
    """
    运行3D实验
    
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
    print(f"3D Experiment: k={k_colors}, model={model_type}")
    print('='*60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 创建数据生成器
    generator = PointPairGenerator(
        dim=3,
        batch_size=batch_size,
        space_range=(-3.0, 3.0),
        device=device,
        seed=42
    )
    
    # 创建模型
    if model_type == "mlp":
        model = MLPColorMapper(
            input_dim=3,
            num_colors=k_colors,
            hidden_dims=hidden_dims,
            activation="relu",
            use_batch_norm=True,
            dropout_rate=0.1,
            use_fourier=True,
            fourier_features=256,
            fourier_sigma=5.0
        )
    elif model_type == "residual":
        model = ResidualColorNet(
            input_dim=3,
            num_colors=k_colors,
            hidden_dim=256,
            num_blocks=4,
            activation="relu",
            dropout_rate=0.1,
            use_fourier=True,
            fourier_features=256,
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
        checkpoint_dir=os.path.join(output_dir, f"checkpoints_3d_k{k_colors}"),
        log_dir=os.path.join(output_dir, f"logs_3d_k{k_colors}"),
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
    
    print(f"Training {model_type.upper()} model with {k_colors} colors in 3D...")
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
        save_path=os.path.join(output_dir, f"training_history_3d_k{k_colors}.png")
    )
    
    # 可视化3D染色方案
    fig_3d = visualizer.visualize_3d_coloring(
        model, k_colors,
        resolution=50,
        space_range=(-2, 2),
        save_path=os.path.join(output_dir, f"3d_coloring_k{k_colors}.png"),
        projection_slice=0.0
    )
    
    # 可视化违反约束的点对
    fig_violations = validator.visualize_violations(
        model,
        save_path=os.path.join(output_dir, f"violations_3d_k{k_colors}.png"),
        num_samples=10000,
        max_display=50
    )
    
    # 分析颜色分布
    color_analysis = validator.analyze_color_distribution(model, num_samples=10000)
    
    # 保存结果
    results = {
        'dimension': 3,
        'k_colors': k_colors,
        'model_type': model_type,
        'final_violation_rate': final_metrics['violation_rate'],
        'best_violation_rate': trainer.best_loss,
        'final_metrics': final_metrics,
        'all_metrics': all_metrics,
        'color_analysis': color_analysis,
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
    results_file = os.path.join(output_dir, f"results_3d_k{k_colors}.json")
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
    
    print(f"\n3D Experiment completed for k={k_colors}")
    print(f"Final violation rate: {final_metrics['violation_rate']:.2f}%")
    print(f"Best violation rate: {trainer.best_loss:.2f}%")
    print(f"Results saved to: {results_file}")
    
    return results


def run_3d_color_sweep(color_range: List[int] = [4, 5, 6, 7, 8, 9, 10],
                      model_type: str = "mlp",
                      epochs: int = 1500,
                      output_dir: str = "./results_3d_sweep") -> Dict[int, Dict[str, Any]]:
    """
    运行3D颜色数扫描实验
    
    Args:
        color_range: 要测试的颜色数列表
        model_type: 模型类型
        epochs: 每个实验的训练轮数
        output_dir: 输出目录
        
    Returns:
        all_results: 所有实验结果字典
    """
    
    print("=" * 80)
    print("3D Color Number Sweep Experiment")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for k in color_range:
        print(f"\nTesting k = {k}")
        print("-" * 40)
        
        try:
            results = run_3d_experiment(
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
        analyze_3d_results(all_results, output_dir)
    
    return all_results


def analyze_3d_results(results: Dict[int, Dict[str, Any]], output_dir: str):
    """分析3D实验结果"""
    
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
    ax1.set_title('3D: Violation Rate vs Number of Colors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 使用的颜色数
    ax2 = axes[0, 1]
    used_colors = [results[k]['all_metrics']['color']['num_used_colors'] for k in k_values]
    ax2.bar(k_values, used_colors, alpha=0.7)
    ax2.plot(k_values, k_values, 'r--', label='Ideal (all colors used)')
    ax2.set_xlabel('Available colors (k)')
    ax2.set_ylabel('Actually used colors')
    ax2.set_title('Color Usage Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 熵和置信度
    ax3 = axes[1, 0]
    entropies = [results[k]['all_metrics']['color']['mean_entropy'] for k in k_values]
    confidences = [results[k]['all_metrics']['color']['mean_confidence'] for k in k_values]
    
    ax3_ent = ax3.twinx()
    
    line1, = ax3.plot(k_values, entropies, 'g-', linewidth=2, label='Mean Entropy')
    line2, = ax3_ent.plot(k_values, confidences, 'b-', linewidth=2, label='Mean Confidence')
    
    ax3.set_xlabel('Number of colors (k)')
    ax3.set_ylabel('Entropy', color='g')
    ax3_ent.set_ylabel('Confidence', color='b')
    ax3.set_title('Model Certainty Metrics')
    ax3.grid(True, alpha=0.3)
    
    # 合并图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    
    # 4. 总体评分
    ax4 = axes[1, 1]
    overall_scores = [results[k]['all_metrics']['overall_score'] for k in k_values]
    ax4.plot(k_values, overall_scores, 'm-', linewidth=2, marker='s', markersize=8)
    ax4.set_xlabel('Number of colors (k)')
    ax4.set_ylabel('Overall Score')
    ax4.set_title('Overall Model Performance')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.suptitle('3D Hadwiger-Nelson Problem Analysis', fontsize=16)
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, "3d_results_summary.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n3D results summary saved to: {plot_path}")
    
    # 计算估计的下界
    feasible_k = []
    for k in k_values:
        if results[k]['final_violation_rate'] < 1.0:  # 1%阈值
            feasible_k.append(k)
    
    if feasible_k:
        estimated_lower_bound = min(feasible_k)
        print(f"\nEstimated lower bound for χ(ℝ³): ≥ {estimated_lower_bound}")
        print(f"Feasible k values: {sorted(feasible_k)}")
        
        # 保存下界估计
        bound_file = os.path.join(output_dir, "estimated_lower_bound.txt")
        with open(bound_file, 'w') as f:
            f.write(f"Estimated lower bound for χ(ℝ³): ≥ {estimated_lower_bound}\n")
            f.write(f"Based on experiments with k = {k_values}\n")
            f.write(f"Feasible k values (violation rate < 1%): {sorted(feasible_k)}\n")
        
        print(f"Lower bound estimate saved to: {bound_file}")
    else:
        print("\nNo feasible k found in tested range")
    
    return fig


if __name__ == "__main__":
    # 运行单个实验
    # results = run_3d_experiment(k_colors=7, output_dir="./results_3d_single")
    
    # 运行颜色数扫描实验
    all_results = run_3d_color_sweep(
        color_range=[4, 5, 6, 7, 8, 9, 10],
        epochs=1500,
        output_dir="./results_3d_sweep"
    )