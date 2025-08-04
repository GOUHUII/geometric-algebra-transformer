# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
"""OMAD-6 医学图像分割数据集加载器"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OMAD6Dataset(Dataset):
    """OMAD-6 医学图像分割数据集
    
    这是一个基于COCO格式的医学图像分割数据集，包含多个解剖结构的分割标注。
    
    Parameters
    ----------
    data_dir : str or Path
        数据集根目录路径
    split : str
        数据集分割，可选 'train2017', 'val2017', 'test2017'
    image_size : int, optional
        图像大小，默认为256
    max_items : int, optional
        每张图像最大像素数量（用于内存控制），默认为1024
    subsample : float, optional
        子采样率，用于快速实验，默认为None（使用全部数据）
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train2017",
        image_size: int = 256,
        max_items: int = 1024,
        subsample: float = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.max_items = max_items
        
        # 定义图像变换
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        # 加载标注文件
        annotation_file = self.data_dir / "annotations" / f"instances_{split}.json"
        if not annotation_file.exists():
            raise FileNotFoundError(f"标注文件不存在: {annotation_file}")
            
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
        
        # 构建图像ID到文件名的映射
        self.image_info = {img['id']: img for img in self.coco_data['images']}
        
        # 构建图像ID到标注的映射
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_annotations:
                self.image_annotations[image_id] = []
            self.image_annotations[image_id].append(ann)
        
        # 获取有标注的图像ID列表
        self.image_ids = list(self.image_annotations.keys())
        
        # 子采样
        if subsample is not None:
            num_samples = max(1, int(len(self.image_ids) * subsample))
            self.image_ids = self.image_ids[:num_samples]
        
        print(f"加载 {split} 数据集，包含 {len(self.image_ids)} 张图像")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取一个数据样本
        
        Returns
        -------
        pixel_data : torch.Tensor with shape (num_pixels, 5)
            像素数据：[x坐标, y坐标, R, G, B]
        segmentation_labels : torch.Tensor with shape (num_pixels,)
            分割标签，0表示背景，1-N表示不同的解剖结构
        """
        image_id = self.image_ids[idx]
        
        # 加载图像
        image_info = self.image_info[image_id]
        image_path = self.data_dir / self.split / image_info['file_name']
        
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 打开并转换图像
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)  # (3, H, W)
        
        # 创建分割掩码
        segmentation_mask = self._create_segmentation_mask(image_id, image.shape[1:])
        
        # 转换为像素级数据
        pixel_data, labels = self._image_to_pixels(image, segmentation_mask)
        
        return pixel_data, labels
    
    def _create_segmentation_mask(self, image_id: int, image_shape: Tuple[int, int]) -> np.ndarray:
        """创建分割掩码"""
        H, W = image_shape
        mask = np.zeros((H, W), dtype=np.int64)
        
        # 获取该图像的所有标注
        annotations = self.image_annotations.get(image_id, [])
        
        for i, ann in enumerate(annotations):
            if 'segmentation' in ann and ann['segmentation']:
                # 简化处理：将每个标注区域标记为不同的类别
                # 在实际应用中，可能需要更复杂的掩码生成逻辑
                category_id = ann['category_id']
                
                # 这里简化处理，根据边界框创建掩码
                # 实际应用中应该使用多边形分割信息
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                
                # 将边界框坐标缩放到当前图像尺寸
                x = int(x * W / self.image_info[image_id]['width'])
                y = int(y * H / self.image_info[image_id]['height'])
                w = int(w * W / self.image_info[image_id]['width'])
                h = int(h * H / self.image_info[image_id]['height'])
                
                # 确保坐标在有效范围内
                x = max(0, min(x, W-1))
                y = max(0, min(y, H-1))
                w = max(1, min(w, W-x))
                h = max(1, min(h, H-y))
                
                # 使用类别ID作为标签（限制在合理范围内）
                label = min(category_id, 10)  # 最多10个类别
                mask[y:y+h, x:x+w] = label
        
        return mask
    
    def _image_to_pixels(self, image: torch.Tensor, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """将图像转换为像素级数据"""
        C, H, W = image.shape
        
        # 创建坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )
        
        # 重塑数据
        x_coords = x_coords.flatten()  # (H*W,)
        y_coords = y_coords.flatten()  # (H*W,)
        rgb_values = image.permute(1, 2, 0).flatten(0, 1)  # (H*W, 3)
        labels = torch.from_numpy(mask).flatten()  # (H*W,)
        
        # 组合像素数据：[x, y, R, G, B]
        pixel_data = torch.stack([
            x_coords, y_coords,
            rgb_values[:, 0], rgb_values[:, 1], rgb_values[:, 2]
        ], dim=1)  # (H*W, 5)
        
        # 随机采样以控制内存使用
        total_pixels = pixel_data.shape[0]
        if total_pixels > self.max_items:
            indices = torch.randperm(total_pixels)[:self.max_items]
            pixel_data = pixel_data[indices]
            labels = labels[indices]
        
        return pixel_data, labels


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """自定义批处理函数，处理不同数量像素的图像"""
    pixel_data_list = []
    labels_list = []
    
    for pixel_data, labels in batch:
        pixel_data_list.append(pixel_data)
        labels_list.append(labels)
    
    # 将所有像素数据连接起来
    all_pixel_data = torch.cat(pixel_data_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    
    # 添加批次维度
    all_pixel_data = all_pixel_data.unsqueeze(0)  # (1, total_pixels, 5)
    all_labels = all_labels.unsqueeze(0)  # (1, total_pixels)
    
    return all_pixel_data, all_labels