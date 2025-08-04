# Copyright (c) 2024 Qualcomm Technologies, Inc.  
# All rights reserved.
"""OMAD-6图像分割实验的实用工具函数"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from pathlib import Path


def visualize_segmentation_results(
    image: torch.Tensor,
    ground_truth: torch.Tensor, 
    prediction: torch.Tensor,
    save_path: Optional[Path] = None,
    title: str = "分割结果对比"
) -> None:
    """可视化分割结果
    
    Parameters
    ----------
    image : torch.Tensor with shape (3, H, W) or (H, W, 3)
        原始图像
    ground_truth : torch.Tensor with shape (H, W)
        真实分割掩码
    prediction : torch.Tensor with shape (H, W)  
        预测分割掩码
    save_path : Path, optional
        保存路径，如果提供则保存图像
    title : str
        图像标题
    """
    # 确保数据在CPU上
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    # 调整图像格式
    if image.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
        image = np.transpose(image, (1, 2, 0))
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title("原始图像")
    axes[0].axis('off')
    
    # 真实分割
    im1 = axes[1].imshow(ground_truth, cmap='tab10', vmin=0, vmax=10)
    axes[1].set_title("真实分割")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 预测分割
    im2 = axes[2].imshow(prediction, cmap='tab10', vmin=0, vmax=10)
    axes[2].set_title("预测分割")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_iou_metrics(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> dict:
    """计算IoU指标
    
    Parameters
    ----------
    predictions : torch.Tensor
        预测掩码
    targets : torch.Tensor
        真实掩码
    num_classes : int
        类别数量
        
    Returns
    -------
    metrics : dict
        包含各类IoU和平均IoU的字典
    """
    device = predictions.device
    
    # 展平张量
    pred_flat = predictions.view(-1)
    target_flat = targets.view(-1)
    
    # 忽略无效标签
    valid_mask = (target_flat >= 0) & (target_flat < num_classes)
    pred_flat = pred_flat[valid_mask]
    target_flat = target_flat[valid_mask]
    
    metrics = {}
    ious = []
    
    for class_id in range(num_classes):
        # 当前类别的预测和真实掩码
        pred_mask = (pred_flat == class_id)
        target_mask = (target_flat == class_id)
        
        # 计算交集和并集
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union > 0:
            iou = intersection / union
        else:
            # 如果该类别不存在且预测也没有预测该类别，IoU为1
            iou = torch.tensor(1.0, device=device) if pred_mask.sum() == 0 else torch.tensor(0.0, device=device)
        
        metrics[f'iou_class_{class_id}'] = iou.item()
        ious.append(iou.item())
    
    # 平均IoU
    metrics['mean_iou'] = np.mean(ious)
    
    return metrics


def convert_pixel_coords_to_image(
    pixel_data: torch.Tensor,
    predictions: torch.Tensor,
    image_size: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将像素级预测转换为图像格式
    
    Parameters
    ----------
    pixel_data : torch.Tensor with shape (num_pixels, 5)
        像素数据 [x, y, R, G, B]，其中坐标范围为[-1, 1]
    predictions : torch.Tensor with shape (num_pixels,)
        像素级预测
    image_size : int
        目标图像大小
        
    Returns
    -------
    image : torch.Tensor with shape (3, H, W)
        重构的RGB图像
    seg_mask : torch.Tensor with shape (H, W)
        分割掩码
    """
    device = pixel_data.device
    
    # 提取坐标和RGB
    coords = pixel_data[:, :2]  # (num_pixels, 2)，范围[-1, 1]
    rgb = pixel_data[:, 2:5]    # (num_pixels, 3)
    
    # 将坐标从[-1, 1]转换到[0, image_size-1]
    coords_scaled = ((coords + 1) * (image_size - 1) / 2).long()
    coords_scaled = torch.clamp(coords_scaled, 0, image_size - 1)
    
    # 初始化图像和掩码
    image = torch.zeros(3, image_size, image_size, device=device)
    seg_mask = torch.zeros(image_size, image_size, dtype=torch.long, device=device)
    
    # 填充像素值
    x_coords = coords_scaled[:, 0]
    y_coords = coords_scaled[:, 1]
    
    # 使用scatter操作填充图像
    for c in range(3):
        image[c, y_coords, x_coords] = rgb[:, c]
    
    # 填充分割掩码
    seg_mask[y_coords, x_coords] = predictions
    
    return image, seg_mask


def save_experiment_summary(
    metrics: dict,
    config: dict,
    save_path: Path
) -> None:
    """保存实验总结
    
    Parameters
    ----------
    metrics : dict
        实验指标
    config : dict
        实验配置
    save_path : Path
        保存路径
    """
    summary = {
        "实验配置": config,
        "最终指标": metrics,
        "关键结果": {
            "平均IoU": metrics.get('mean_iou', 0.0),
            "整体准确率": metrics.get('overall_accuracy', 0.0),
            "损失": metrics.get('loss', 0.0)
        }
    }
    
    # 保存为文本文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=== OMAD-6 图像分割实验总结 ===\n\n")
        
        f.write("关键结果:\n")
        for key, value in summary["关键结果"].items():
            f.write(f"  {key}: {value:.4f}\n")
        f.write("\n")
        
        f.write("详细指标:\n")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"  {key}: {value:.4f}\n")
        f.write("\n")
        
        f.write("实验配置:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")


class SegmentationLogger:
    """分割实验的日志记录器"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history = []
        
    def log_metrics(self, step: int, metrics: dict, tag: str = "train"):
        """记录指标"""
        entry = {
            "step": step,
            "tag": tag,
            "timestamp": np.datetime64('now'),
            **metrics
        }
        self.metrics_history.append(entry)
        
        # 打印关键指标
        if tag == "val":
            print(f"Step {step} - Val mIoU: {metrics.get('mean_iou', 0):.4f}, "
                  f"Acc: {metrics.get('overall_accuracy', 0):.4f}")
    
    def save_metrics_history(self):
        """保存指标历史"""
        if self.metrics_history:
            # 这里可以保存为CSV或JSON格式
            # 简化实现，只打印总结
            val_metrics = [m for m in self.metrics_history if m.get('tag') == 'val']
            if val_metrics:
                best_miou = max(val_metrics, key=lambda x: x.get('mean_iou', 0))
                print(f"\n=== 最佳验证结果 ===")
                print(f"Step: {best_miou['step']}")
                print(f"mIoU: {best_miou.get('mean_iou', 0):.4f}")
                print(f"Accuracy: {best_miou.get('overall_accuracy', 0):.4f}")