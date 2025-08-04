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


class OMAD6Experiment(BaseExperiment):
    """OMAD-6图像分割实验管理器
    
    管理医学图像分割任务的训练、验证和测试流程。
    包括损失计算、指标评估、数据加载等功能。
    
    Parameters
    ----------
    cfg : OmegaConf
        实验配置，参考config文件夹中的示例
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # 分割损失函数
        self._ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        
        # 用于计算IoU等指标
        self.num_classes = cfg.data.get('num_classes', 11)
    
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
    
    def _load_dataloader(self, tag: str) -> DataLoader:
        """创建数据加载器
        
        Parameters
        ----------
        tag : str
            数据集标签
            
        Returns
        -------
        dataloader : DataLoader
            PyTorch数据加载器
        """
        dataset = self._load_dataset(tag)
        
        # 根据标签确定批次大小和是否打乱
        if tag == "train":
            batch_size = self.cfg.training.batchsize
            shuffle = True
        else:
            batch_size = self.cfg.training.get('eval_batchsize', self.cfg.training.batchsize)
            shuffle = False
        
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
        
        # 计算交叉熵损失
        ce_loss = self._ce_criterion(logits_flat, labels_flat)
        
        # 总损失
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
    
    def _compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """计算详细的分割指标
        
        Parameters
        ----------
        predictions : torch.Tensor
            预测logits
        targets : torch.Tensor  
            真实标签
            
        Returns
        -------
        metrics : dict
            包含IoU、准确率等指标的字典
        """
        with torch.no_grad():
            # 获取预测类别
            pred_classes = torch.argmax(predictions, dim=-1)  # (batch_size, num_pixels)
            
            batch_size = targets.shape[0]
            metrics = {}
            
            # 计算整体准确率
            correct = (pred_classes == targets).float()
            valid_mask = (targets != -1)
            if valid_mask.sum() > 0:
                overall_accuracy = correct[valid_mask].mean().item()
            else:
                overall_accuracy = 0.0
            
            metrics['overall_accuracy'] = overall_accuracy
            
            # 计算每个类别的IoU
            class_ious = []
            for class_id in range(self.num_classes):
                # 预测为该类别的像素
                pred_mask = (pred_classes == class_id)
                # 真实为该类别的像素
                target_mask = (targets == class_id)
                
                # 计算交集和并集
                intersection = (pred_mask & target_mask & valid_mask).sum().float()
                union = ((pred_mask | target_mask) & valid_mask).sum().float()
                
                if union > 0:
                    iou = intersection / union
                    class_ious.append(iou.item())
                    metrics[f'iou_class_{class_id}'] = iou.item()
                else:
                    # 如果该类别在真实标签中不存在，且预测也没有预测该类别，则IoU为1
                    if pred_mask.sum() == 0:
                        class_ious.append(1.0)
                        metrics[f'iou_class_{class_id}'] = 1.0
                    else:
                        class_ious.append(0.0)
                        metrics[f'iou_class_{class_id}'] = 0.0
            
            # 平均IoU
            if class_ious:
                metrics['mean_iou'] = sum(class_ious) / len(class_ious)
            else:
                metrics['mean_iou'] = 0.0
            
            return metrics
    
    def _evaluate_model(self, tag: str) -> Dict[str, float]:
        """评估模型性能
        
        Parameters
        ----------
        tag : str
            评估数据集标签
            
        Returns
        -------
        metrics : dict
            评估指标
        """
        self.model.eval()
        dataloader = self._load_dataloader(tag)
        
        total_loss = 0.0
        all_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                # 移动到设备
                pixel_data, labels = batch_data
                pixel_data = pixel_data.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                loss, batch_metrics = self._forward(pixel_data, labels)
                
                # 累积损失和指标
                total_loss += loss.item()
                for key, value in batch_metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
                
                # 计算详细指标
                logits = self.model(pixel_data)
                detailed_metrics = self._compute_metrics(logits, labels)
                for key, value in detailed_metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
                
                num_batches += 1
        
        # 平均指标
        avg_metrics = {}
        avg_metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        
        for key, values in all_metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics