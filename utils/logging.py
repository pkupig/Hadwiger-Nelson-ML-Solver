import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime
import torch


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, 
                 log_dir: str = "./logs",
                 experiment_name: Optional[str] = None,
                 level: int = logging.INFO):
        
        # 创建日志目录
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成实验名称
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"experiment_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # 创建实验目录
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(level)
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        log_file = self.experiment_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(console_formatter)
        self.logger.addHandler(file_handler)
        
        # 保存配置
        self.config_file = self.experiment_dir / "config.json"
        self.metrics_file = self.experiment_dir / "metrics.json"
        
        # 初始化指标
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'violation_rate': [],
            'learning_rate': [],
            'timestamps': []
        }
        
        self.start_time = time.time()
        
        self.logger.info(f"Experiment started: {self.experiment_name}")
        self.logger.info(f"Log directory: {self.experiment_dir}")
    
    def log_config(self, config: Dict[str, Any]):
        """记录实验配置"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info("Experiment configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_metrics(self, 
                   epoch: int,
                   train_loss: Optional[float] = None,
                   val_loss: Optional[float] = None,
                   violation_rate: Optional[float] = None,
                   learning_rate: Optional[float] = None):
        """记录训练指标"""
        
        timestamp = time.time() - self.start_time
        
        if train_loss is not None:
            self.metrics['train_loss'].append({'epoch': epoch, 'value': train_loss, 'time': timestamp})
        
        if val_loss is not None:
            self.metrics['val_loss'].append({'epoch': epoch, 'value': val_loss, 'time': timestamp})
        
        if violation_rate is not None:
            self.metrics['violation_rate'].append({'epoch': epoch, 'value': violation_rate, 'time': timestamp})
        
        if learning_rate is not None:
            self.metrics['learning_rate'].append({'epoch': epoch, 'value': learning_rate, 'time': timestamp})
        
        self.metrics['timestamps'].append({'epoch': epoch, 'time': timestamp})
        
        # 保存到文件
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_message(self, message: str, level: str = "info"):
        """记录消息"""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)
    
    def log_checkpoint(self, 
                      model, 
                      optimizer, 
                      epoch: int,
                      loss: float,
                      additional_info: Optional[Dict] = None):
        """保存检查点"""
        
        checkpoint_dir = self.experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch:04d}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': time.time()
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def log_model_summary(self, model):
        """记录模型摘要"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model Summary:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        # 记录每层信息
        self.logger.info("  Layer details:")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 只记录叶子模块
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    self.logger.info(f"    {name}: {num_params:,} parameters")
    
    def log_experiment_end(self, final_metrics: Dict[str, Any]):
        """记录实验结束"""
        elapsed_time = time.time() - self.start_time
        
        self.logger.info(f"Experiment completed: {self.experiment_name}")
        self.logger.info(f"Total time: {elapsed_time:.2f} seconds")
        self.logger.info("Final metrics:")
        
        for key, value in final_metrics.items():
            self.logger.info(f"  {key}: {value}")
        
        # 保存最终指标
        final_file = self.experiment_dir / "final_results.json"
        final_results = {
            'experiment_name': self.experiment_name,
            'total_time': elapsed_time,
            'final_metrics': final_metrics,
            'all_metrics': self.metrics
        }
        
        with open(final_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        self.logger.info(f"Final results saved to: {final_file}")
    
    def get_metrics_plot_data(self) -> Dict[str, Any]:
        """获取用于绘制图表的指标数据"""
        return self.metrics


class TensorBoardLogger:
    """TensorBoard日志记录器包装器"""
    
    def __init__(self, log_dir: str = "./runs", experiment_name: Optional[str] = None):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_available = True
            
            if experiment_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_name = f"experiment_{timestamp}"
            
            self.log_dir = Path(log_dir) / experiment_name
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            self.writer = SummaryWriter(str(self.log_dir))
            
            print(f"TensorBoard logging enabled: {self.log_dir}")
            print(f"To view: tensorboard --logdir={log_dir}")
            
        except ImportError:
            self.tensorboard_available = False
            print("TensorBoard not available. Install with: pip install tensorboard")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量值"""
        if self.tensorboard_available:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """记录多个标量值"""
        if self.tensorboard_available:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """记录直方图"""
        if self.tensorboard_available:
            self.writer.add_histogram(tag, values, step)
    
    def log_graph(self, model, input_tensor):
        """记录计算图"""
        if self.tensorboard_available:
            self.writer.add_graph(model, input_tensor)
    
    def log_embedding(self, embeddings, metadata=None, label_img=None, step: int = 0):
        """记录嵌入向量"""
        if self.tensorboard_available:
            self.writer.add_embedding(embeddings, metadata=metadata, label_img=label_img, global_step=step)
    
    def close(self):
        """关闭写入器"""
        if self.tensorboard_available:
            self.writer.close()


def setup_logging(log_dir: str = "./logs", 
                 experiment_name: Optional[str] = None,
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG) -> ExperimentLogger:
    """设置日志记录"""
    
    return ExperimentLogger(log_dir, experiment_name, console_level)


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志记录器"""
    return logging.getLogger(name)


class ProgressLogger:
    """进度记录器"""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1, metrics: Optional[Dict] = None):
        """更新进度"""
        self.current += n
        
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0
        
        progress = self.current / self.total * 100
        
        message = f"{self.description}: {self.current}/{self.total} ({progress:.1f}%)"
        message += f" - {rate:.1f} it/s"
        message += f" - ETA: {remaining:.1f}s"
        
        if metrics:
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            message += f" | {metric_str}"
        
        print(f"\r{message}", end="", flush=True)
    
    def close(self):
        """关闭进度记录器"""
        elapsed = time.time() - self.start_time
        print(f"\n{self.description} completed in {elapsed:.2f} seconds")