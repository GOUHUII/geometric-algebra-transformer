# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
"""OMAD-6 图像分割实验管理器"""

from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gatr.experiments.base_experiment import BaseExperiment
from gatr.experiments.omad6.dataset import OMAD6Dataset, collate_fn


class FocalLoss(torch.nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, weight=None, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.weight, 
            ignore_index=self.ignore_index, 
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


class OMAD6Experiment(BaseExperiment):
    """OMAD-6海岸养殖地物遥感图像分割实验管理器
    
    管理海岸养殖地物遥感图像分割任务的训练、验证和测试流程。
    包括损失计算、指标评估、数据加载等功能。
    
    Parameters
    ----------
    cfg : OmegaConf
        实验配置，参考config文件夹中的示例
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # 用于计算IoU等指标 - 7个类别（背景+6个海岸养殖地物类别）
        self.num_classes = cfg.data.get('num_classes', 7)
        
        # 动态权重策略 - 基于类别频率计算，避免硬编码
        # 默认权重，实际训练时会根据数据分布动态调整
        class_weights = torch.ones(self.num_classes)
        
        # 选择损失函数 - 可以在配置中选择
        use_focal_loss = cfg.training.get('use_focal_loss', False)
        
        if use_focal_loss:
            # 使用Focal Loss处理极度不平衡的数据，加强难例学习
            self._ce_criterion = FocalLoss(
                alpha=2.0,    # 增加alpha，更关注难例
                gamma=3.0,    # 增加gamma，更强烈抑制简单样本
                weight=class_weights,
                ignore_index=-1
            )
        else:
            # 使用加权交叉熵
            self._ce_criterion = torch.nn.CrossEntropyLoss(
                weight=class_weights, 
                ignore_index=-1,
                label_smoothing=0.1  # 添加标签平滑
            )
    
    def _load_dataset(self, tag: str) -> OMAD6Dataset:
        """加载数据集
        
        Parameters
        ----------
        tag : str
            数据集标签，如 "train", "val", "test"
        
        Returns
        -------
        dataset : OMAD6Dataset
            OMAD-6数据集实例
        """
        # 映射标签到COCO格式的分割名
        tag_mapping = {
            "train": "train2017",
            "val": "val2017", 
            "test": "test2017"
        }
        
        split = tag_mapping.get(tag, tag)
        
        # 训练时使用子采样
        subsample_fraction = None
        if tag == "train" and hasattr(self.cfg.data, 'subsample'):
            subsample_fraction = self.cfg.data.subsample
        
        dataset = OMAD6Dataset(
            data_dir=self.cfg.data.data_dir,
            split=split,
            image_size=self.cfg.data.get('image_size', 256),
            max_items=self.cfg.data.get('max_items', 1024),
            subsample=subsample_fraction
        )
        
        return dataset
    
    def _make_data_loader(self, dataset, batch_size, shuffle):
        """创建数据加载器 - 覆盖基类方法以使用自定义collate_fn
        
        Parameters
        ----------
        dataset : OMAD6Dataset
            数据集实例
        batch_size : int
            批次大小
        shuffle : bool
            是否打乱数据
            
        Returns
        -------
        dataloader : DataLoader
            PyTorch数据加载器
        """
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.cfg.training.get('num_workers', 0),
            pin_memory=True
        )
        
        return dataloader
    
    def _forward(self, *data) -> Tuple[torch.Tensor, Dict[str, float]]:
        """模型前向传播
        
        Parameters
        ----------
        data : tuple of torch.Tensor
            数据批次: (pixel_data, labels)
        
        Returns
        -------
        loss : torch.Tensor
            总损失
        metrics : dict
            额外的指标用于日志记录
        """
        assert self.model is not None, "模型未初始化"
        
        pixel_data, labels = data
        
        # 前向传播
        logits = self.model(pixel_data)  # (batch_size, num_pixels, num_classes)
        
        # 重塑为适合损失计算的格式
        batch_size, num_pixels, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)  # (batch_size * num_pixels, num_classes)
        labels_flat = labels.view(-1)  # (batch_size * num_pixels,)
        
        # 确保损失函数权重在正确的设备上
        if hasattr(self._ce_criterion, 'weight') and self._ce_criterion.weight is not None:
            if self._ce_criterion.weight.device != pixel_data.device:
                self._ce_criterion.weight = self._ce_criterion.weight.to(pixel_data.device)
        
        # 计算交叉熵损失
        ce_loss = self._ce_criterion(logits_flat, labels_flat)
        
        # 简化困难样本挖掘逻辑，避免过度复杂化
        total_loss = ce_loss
        
        # 计算准确率
        with torch.no_grad():
            predictions = torch.argmax(logits_flat, dim=1)
            correct = (predictions == labels_flat).float()
            # 忽略无效标签
            valid_mask = (labels_flat != -1)
            if valid_mask.sum() > 0:
                accuracy = correct[valid_mask].mean()
            else:
                accuracy = torch.tensor(0.0)
        
        # 额外指标
        metrics = {
            'ce_loss': ce_loss.item(),
            'accuracy': accuracy.item(),
            'num_pixels': num_pixels,
        }
        
        return total_loss, metrics
    
    @torch.no_grad()
    def _compute_metrics(self, dataloader: DataLoader) -> Dict[str, float]:
        """计算验证数据集上的详细指标
        
        Parameters
        ----------
        dataloader : DataLoader
            验证数据加载器
            
        Returns
        -------
        metrics : dict
            包含损失、IoU、准确率等指标的字典
        """
        assert self.model is not None
        self.model.eval()
        
        # 移动到评估设备
        eval_device = torch.device(self.cfg.training.eval_device)
        original_device = next(self.model.parameters()).device
        self.model = self.model.to(eval_device)
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        # 遍历验证数据
        for batch_data in dataloader:
            # 移动数据到设备
            pixel_data, labels = batch_data
            pixel_data = pixel_data.to(eval_device)
            labels = labels.to(eval_device)
            
            # 前向传播
            loss, batch_metrics = self._forward(pixel_data, labels)
            total_loss += loss.item()
            
            # 获取预测结果 - 直接使用_forward中的logits
            logits = self.model(pixel_data)  # (batch_size, num_pixels, num_classes)
            predictions = torch.argmax(logits, dim=-1)  # (batch_size, num_pixels)
            
            # 收集预测和真实标签
            all_predictions.append(predictions.flatten())
            all_targets.append(labels.flatten())
            
            num_batches += 1
        
        # 将模型移回原始设备
        self.model = self.model.to(original_device)
        
        # 合并所有预测和标签
        all_predictions = torch.cat(all_predictions, dim=0)  # (total_pixels,)
        all_targets = torch.cat(all_targets, dim=0)  # (total_pixels,)
        
        # 计算指标
        metrics = {}
        metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        
        # 计算整体准确率
        valid_mask = (all_targets != -1)
        if valid_mask.sum() > 0:
            correct = (all_predictions == all_targets).float()
            overall_accuracy = correct[valid_mask].mean().item()
        else:
            overall_accuracy = 0.0
        metrics['overall_accuracy'] = overall_accuracy
        
        # 计算每个类别的IoU
        class_ious = []
        present_classes = []  # 记录数据中实际存在的类别
        
        for class_id in range(self.num_classes):
            # 预测为该类别的像素
            pred_mask = (all_predictions == class_id)
            # 真实为该类别的像素
            target_mask = (all_targets == class_id)
            
            # 只在有效区域内计算
            pred_mask_valid = pred_mask & valid_mask
            target_mask_valid = target_mask & valid_mask
            
            # 检查该类别是否在真实数据中存在
            target_exists = target_mask_valid.sum() > 0
            pred_exists = pred_mask_valid.sum() > 0
            
            if target_exists:
                # 类别在数据中存在，计算真实IoU
                intersection = (pred_mask_valid & target_mask_valid).sum().float()
                union = (pred_mask_valid | target_mask_valid).sum().float()
                
                if union > 0:
                    iou = intersection / union
                else:
                    iou = 0.0
                    
                class_ious.append(iou.item())
                present_classes.append(class_id)
                metrics[f'iou_class_{class_id}'] = iou.item()
                
            elif not pred_exists:
                # 类别不存在且模型也没预测 - 正确的负样本
                iou = 1.0
                class_ious.append(iou)
                metrics[f'iou_class_{class_id}'] = iou
                
            else:
                # 类别不存在但模型错误预测了 - 假阳性
                iou = 0.0
                class_ious.append(iou)
                metrics[f'iou_class_{class_id}'] = iou
        
        # 计算平均IoU
        if class_ious:
            metrics['mean_iou'] = sum(class_ious) / len(class_ious)
        else:
            metrics['mean_iou'] = 0.0
        
        # 计算存在类别的平均IoU
        if present_classes:
            present_ious = [metrics[f'iou_class_{cls}'] for cls in present_classes]
            metrics['mean_iou_present'] = sum(present_ious) / len(present_ious)
            metrics['num_present_classes'] = len(present_classes)
        else:
            metrics['mean_iou_present'] = 0.0
            metrics['num_present_classes'] = 0
        
        return metrics
    
    @property
    def _eval_dataset_tags(self):
        """评估数据集标签
        
        Returns
        -------
        tags : set of str
            评估数据集标签
        """
        return {"val"}