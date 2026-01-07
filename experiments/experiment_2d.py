"""
2D Hadwiger-Nelson实验
验证已知结果：χ(ℝ²) ∈ [4, 7]
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator import PointPairGenerator
from models.mlp_mapper import MLPColorMapper
from losses.constraint_loss import ConstraintLoss
from training.trainer import TrainingConfig, HadwigerNelsonTrainer


def run_2d_experiments():
    """运行2D实验"""
    print("=" * 80)
    print("2D Hadwiger-Nelson Problem Experiments")
    print("Known: χ(ℝ²) ∈ [4, 7] (4 ≤ χ ≤ 7)")
    print("=" * 80)
    
    # 测试的颜色数
    colors_to_test = [3, 4, 5, 6, 7, 8]
    
    results = {}
    
    for k in colors_to_test:
        print(f"\n{'='*60}")
        print(f"Testing 2D with k = {k} colors")
        print('='*60)
        
        # 创建组件
        generator = PointPairGenerator(dim=2, batch_size=4096, device='cuda')
        model = MLPColorMapper(input_dim=2, num_colors=k, hidden_dims=[128, 256, 128])
        loss_fn = ConstraintLoss(conflict_weight=1.0, entropy_weight=0.1)
        
        # 训练配置
        config = TrainingConfig(
            epochs=2000,
            batch_size=4096,
            learning_rate=0.001,
            validation_freq=100,
            device='cuda',
            checkpoint_dir=f"./checkpoints_2d_k{k}",
            log_dir=f"./logs_2d_k{k}"
        )
        
        # 训练
        trainer = HadwigerNelsonTrainer(model, generator, loss_fn, config)
        history = trainer.train()
        
        # 最终评估
        final_metrics = trainer.validate(num_samples=50000)
        
        # 保存结果
        results[k] = {
            'final_violation_rate': final_metrics['violation_rate'],
            'best_violation_rate': trainer.best_loss,
            'history': history
        }
        
        print(f"\nResults for k={k}:")
        print(f"  Final violation rate: {final_metrics['violation_rate']:.2f}%")
        print(f"  Best violation rate: {trainer.best_loss:.2f}%")
        
        # 可视化染色方案
        if k in [4, 7]:  # 关键值
            from evaluation.visualizer import HadwigerNelsonVisualizer
            visualizer = HadwigerNelsonVisualizer()
            visualizer.visualize_2d_coloring(
                model, k, 
                save_path=f"./2d_coloring_k{k}.png"
            )
    
    # 分析结果
    print("\n" + "="*80)
    print("2D Experiment Analysis")
    print("="*80)
    
    feasible_k = []
    for k, result in results.items():
        if result['final_violation_rate'] < 1.0:  # 1%阈值
            feasible_k.append(k)
    
    if feasible_k:
        estimated_lower_bound = min(feasible_k)
        print(f"Estimated lower bound for χ(ℝ²): ≥ {estimated_lower_bound}")
        print(f"Feasible k values: {sorted(feasible_k)}")
        
        # 验证已知范围
        if estimated_lower_bound >= 4 and estimated_lower_bound <= 7:
            print("✓ Results consistent with known bounds: χ(ℝ²) ∈ [4, 7]")
        else:
            print("⚠ Results differ from known bounds!")
    else:
        print("No feasible k found in tested range")
    
    # 绘制结果图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 冲突率 vs k
    k_values = sorted(results.keys())
    violation_rates = [results[k]['final_violation_rate'] for k in k_values]
    
    ax1 = axes[0]
    ax1.plot(k_values, violation_rates, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1% threshold')
    ax1.fill_between(k_values, 0, 1.0, alpha=0.1, color='green')
    ax1.axvline(x=4, color='orange', linestyle=':', alpha=0.5, label='Known: χ≥4')
    ax1.axvline(x=7, color='green', linestyle=':', alpha=0.5, label='Known: χ≤7')
    ax1.set_xlabel('Number of colors (k)')
    ax1.set_ylabel('Violation Rate (%)')
    ax1.set_title('2D: Violation Rate vs k')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 训练曲线示例
    ax2 = axes[1]
    if 7 in results:  # 使用k=7的训练曲线
        history = results[7]['history']
        epochs = [h['epoch'] for h in history if h.get('train')]
        losses = [h['train']['total_loss'] for h in history if h.get('train')]
        
        ax2.plot(epochs, losses, 'b-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Training Loss')
        ax2.set_title('Training Curve (2D, k=7)')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    
    plt.suptitle('2D Hadwiger-Nelson Problem Results')
    plt.tight_layout()
    plt.savefig('2d_experiment_results.png', dpi=150, bbox_inches='tight')
    
    print(f"\nResults saved to 2d_experiment_results.png")
    
    return results


if __name__ == "__main__":
    results = run_2d_experiments()