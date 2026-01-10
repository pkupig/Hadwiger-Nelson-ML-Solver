#!/usr/bin/env python3
"""
Hadwiger-Nelson Problem ML Solver
主程序：运行2D、3D、4D实验
"""

import argparse
from logging import config
import os
import sys
from xml.parsers.expat import model
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config.config import load_config
from training.trainer import TrainingConfig, MultiDimensionTrainer, HadwigerNelsonTrainer
from evaluation.visualizer import HadwigerNelsonVisualizer, ResultAnalyzer
from data.generator import PointPairGenerator
from data.hard_examples import HardExampleTrainer
from models.mlp_mapper import MLPColorMapper
from losses.constraint_loss import ConstraintLoss


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Hadwiger-Nelson Problem ML Solver')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--dims', type=int, nargs='+', default=[2, 3, 4],
                       help='要测试的维度')
    parser.add_argument('--colors', type=int, nargs='+', 
                       help='要测试的颜色数（覆盖配置文件）')
    parser.add_argument('--epochs', type=int, default=2000,
                       help='每个实验的训练epoch数')
    parser.add_argument('--batch-size', type=int, default=4096,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'auto'],
                       help='设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='输出目录')
    parser.add_argument('--visualize', action='store_true',
                       help='训练后可视化结果')
    parser.add_argument('--quick-test', action='store_true',
                       help='快速测试模式（减少epoch和颜色数）')
    
    return parser.parse_args()


def setup_environment(args, config):
    """设置环境"""
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定设备
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # 更新配置
    config['project']['device'] = device
    config['project']['seed'] = args.seed
    config['training']['epochs'] = args.epochs
    config['data']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.lr
    
    # 快速测试模式
    if args.quick_test:
        config['training']['epochs'] = 500
        config['experiments']['2d']['colors_to_test'] = [3, 4, 7]
        config['experiments']['3d']['colors_to_test'] = [4, 7, 10]
        config['experiments']['4d']['colors_to_test'] = [5, 8, 12]
    
    # 如果指定了颜色，覆盖配置
    if args.colors:
        for dim in args.dims:
            if dim == 2:
                config['experiments']['2d']['colors_to_test'] = args.colors
            elif dim == 3:
                config['experiments']['3d']['colors_to_test'] = args.colors
            elif dim == 4:
                config['experiments']['4d']['colors_to_test'] = args.colors
    
    return config


def run_single_experiment(dim: int, k: int, config: dict, output_dir: str):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Running experiment: {dim}D with k={k}")
    print('='*60)
    
    import torch
    
    # 创建数据生成器
    generator = PointPairGenerator(
        dim=dim,
        batch_size=config['data']['batch_size'],
        space_range=tuple(config['data']['space_range']),
        device=config['project']['device'],
        seed=config['project']['seed']
    )
    
    # 创建模型 - 确保使用float64
    model = MLPColorMapper(
        input_dim=dim,
        num_colors=k,
        hidden_dims=config['model']['hidden_layers'],
        activation=config['model']['activation'],
        use_batch_norm=config['model']['use_batch_norm'],
        dropout_rate=config['model']['dropout_rate'],
        use_fourier=config['model']['fourier_features']['enabled'],
        fourier_features=config['model']['fourier_features']['num_features'],
        fourier_sigma=config['model']['fourier_features']['sigma']
    )
    
    # 将模型转换为float64
    model = model.to(torch.float64)
    
    # 创建损失函数
    loss_weights = config['training']['loss_weights']
    loss_fn = ConstraintLoss(
        conflict_weight=float(loss_weights['conflict']),
        entropy_weight=float(loss_weights['entropy']),
        uniformity_weight=float(loss_weights['uniformity']),
        spectral_weight=float(loss_weights['spectral'])
    )
    
    # 创建训练配置对象 (TrainingConfig)
    train_config = TrainingConfig(
        epochs=int(config['training']['epochs']),
        batch_size=int(config['data']['batch_size']),
        num_train_pairs=int(config['data'].get('num_train_pairs', 1000000)),
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']), # <--- 这里就是报错的源头
        gradient_clip=float(config['training']['gradient_clip']),
        validation_freq=int(config['evaluation']['validation_freq']),
        save_freq=int(config['evaluation']['save_checkpoint_freq']),
        checkpoint_dir=os.path.join(output_dir, f"checkpoints_{dim}d_k{k}"),
        log_dir=os.path.join(output_dir, f"logs_{dim}d_k{k}"),
        device=config['project']['device'],
        seed=int(config['project']['seed'])
    )
    
    # ---------------------------------------------------------
    # 逻辑修正：动态选择训练器 (Hard Example Mining vs Standard)
    # ---------------------------------------------------------
    use_hard_examples = config['data'].get('hard_examples', {}).get('enabled', False)
    
    if use_hard_examples:
        print(f"\n>>> [System] Activating Hard Example Mining (Active Learning)")
        # 动态导入以避免循环依赖或未使用的导入
        from data.hard_examples import HardExampleTrainer
        
        # 实例化难例挖掘训练器
        trainer = HardExampleTrainer(
            model=model,
            data_generator=generator,
            loss_fn=loss_fn,
            config=train_config,  # 传入 TrainingConfig 对象
            hard_example_ratio=0.3
        )
        
        # 预加载几何难例 (Moser Spindle 等)
        # 这一步非常关键，给模型提供初始的"硬骨头"
        trainer.add_geometric_hard_examples()
        
    else:
        print(f"\n>>> [System] Using Standard Random Sampling Trainer")
        from training.trainer import HadwigerNelsonTrainer
        
        trainer = HadwigerNelsonTrainer(
            model=model,
            data_generator=generator,
            loss_fn=loss_fn,
            config=train_config
        )
    
    # 开始训练
    history = trainer.train()
    
    # 最终评估
    final_metrics = trainer.validate(num_samples=config['evaluation']['num_test_samples'])
    
    # 保存模型
    model_save_path = os.path.join(output_dir, f"model_{dim}d_k{k}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'dim': dim,
        'k_colors': k,
        'final_metrics': final_metrics,
        'config': config
    }, model_save_path)
    
    print(f"\nExperiment completed for {dim}D, k={k}")
    print(f"Final violation rate: {final_metrics['violation_rate']:.2f}%")
    print(f"Model saved to: {model_save_path}")
    
    return {
        'final_violation_rate': final_metrics['violation_rate'],
        'final_loss': final_metrics['validation_loss'],
        'best_violation_rate': trainer.best_loss,
        'history': history,
        'model_path': model_save_path
    }


def main():
    """主函数"""
    import torch
    import numpy as np
    import matplotlib
    
    args = parse_args()
    
    # 设置全局默认dtype为float64
    torch.set_default_dtype(torch.float64)
    print(f"设置全局数据类型为: {torch.get_default_dtype()}")
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置环境
    config = setup_environment(args, config)
    
    print("=" * 80)
    print("Hadwiger-Nelson Problem ML Solver")
    print("=" * 80)
    print(f"Dimensions to test: {args.dims}")
    print(f"Device: {config['project']['device']}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {config['project']['seed']}")
    print(f"Default dtype: {torch.get_default_dtype()}")
    print("=" * 80)
    
    # 运行实验
    all_results = {}
    
    for dim in args.dims:
        dim_key = f"{dim}d"
        if dim_key not in config['experiments']:
            print(f"Warning: No configuration for {dim}D, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Starting experiments for {dim}D")
        print('='*60)
        
        colors_to_test = config['experiments'][dim_key]['colors_to_test']
        all_results[dim] = {}
        
        for k in colors_to_test:
            result = run_single_experiment(dim, k, config, args.output_dir)
            all_results[dim][k] = result
        
        # 可视化该维度的结果
        if args.visualize and dim in [2, 3]:
            print(f"\nVisualizing results for {dim}D...")
            visualizer = HadwigerNelsonVisualizer(device=config['project']['device'])
            
            # 加载最佳模型进行可视化
            best_k = min(all_results[dim].keys(), 
                        key=lambda x: all_results[dim][x]['final_violation_rate'])
            best_model_path = all_results[dim][best_k]['model_path']
            
            checkpoint = torch.load(best_model_path, map_location=config['project']['device'])
            
            # 重新创建模型
            model = MLPColorMapper(
                input_dim=dim,
                num_colors=best_k,
                hidden_dims=config['model']['hidden_layers']
            )
            model = model.to(torch.float64)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(config['project']['device'])
            
            # 可视化
            if dim == 2:
                visualizer.visualize_2d_coloring(
                    model, best_k,
                    resolution=config['evaluation']['grid_resolution']['2d'],
                    save_path=os.path.join(args.output_dir, f"2d_coloring_k{best_k}.png")
                )
            elif dim == 3:
                visualizer.visualize_3d_coloring(
                    model, best_k,
                    resolution=config['evaluation']['grid_resolution']['3d'],
                    save_path=os.path.join(args.output_dir, f"3d_coloring_k{best_k}.png")
                )
    
    # 分析结果
    print("\n" + "=" * 80)
    print("Analyzing results...")
    print("=" * 80)
    
    analyzer = ResultAnalyzer(all_results)
    
    # 生成摘要
    summary = analyzer.generate_summary_table()
    print(summary)
    
    # 保存摘要到文件
    summary_path = os.path.join(args.output_dir, "results_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Summary saved to: {summary_path}")
    
    # 绘制结果图
    fig = analyzer.plot_violation_vs_k(
        save_path=os.path.join(args.output_dir, "violation_vs_k.png")
    )
    
    # 使用MultiDimensionTrainer绘制比较图
    if len(args.dims) > 1:
        print("\nGenerating multi-dimension comparison...")
        
        # 准备颜色配置
        colors_per_dim = {}
        for dim in args.dims:
            dim_key = f"{dim}d"
            if dim_key in config['experiments']:
                colors_per_dim[dim] = config['experiments'][dim_key]['colors_to_test']
        
        # 创建训练配置
        base_config = TrainingConfig(
            epochs=config['training']['epochs'],
            batch_size=config['data']['batch_size'],
            device=config['project']['device']
        )
        
        # 创建多维度训练器（仅用于绘图）
        multi_trainer = MultiDimensionTrainer(
            dims=args.dims,
            colors_per_dim=colors_per_dim,
            base_config=base_config
        )
        
        # 手动设置结果并绘图
        multi_trainer.results = all_results
        fig = multi_trainer.plot_comparison()
    
    print("\n" + "=" * 80)
    print("All experiments completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    # 检查必要的包
    try:
        import torch
        import numpy as np
        import matplotlib
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("Please install requirements: pip install torch numpy matplotlib plotly")
        sys.exit(1)
    
    main()
