# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
"""OMAD-6 医学图像分割数据集加载器"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OMAD6Dataset(Dataset):
    """OMAD-6 海岸养殖地物遥感图像分割数据集
    
    这是一个基于COCO格式的遥感图像分割数据集，用于海岸养殖地物识别和分割。
    包含6个主要地物类别：TCC、DWCC、FRC、LC、RC、BC等养殖设施和地理要素。
    
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
        
        # 智能采样：优先选择有标注的图像，然后补充无标注图像
        annotated_ids = list(self.image_annotations.keys())
        all_ids = list(self.image_info.keys())
        unannotated_ids = [img_id for img_id in all_ids if img_id not in self.image_annotations]
        
        print(f"数据集统计：总图像 {len(all_ids)}，有标注 {len(annotated_ids)}，无标注 {len(unannotated_ids)}")
        
        if subsample is not None:
            num_samples = max(1, int(len(all_ids) * subsample))
            
            # 优先采样策略：70%有标注图像 + 30%无标注图像
            num_annotated = min(len(annotated_ids), int(num_samples * 0.7))
            num_unannotated = num_samples - num_annotated
            
            # 使用随机采样而不是顺序采样，确保类别多样性
            import random
            random.seed(42)  # 确保可重复性
            
            # 确保采样数量不超过可用数量
            actual_annotated = min(num_annotated, len(annotated_ids)) if annotated_ids else 0
            actual_unannotated = min(num_unannotated, len(unannotated_ids)) if unannotated_ids else 0
            
            selected_annotated = random.sample(annotated_ids, actual_annotated) if actual_annotated > 0 else []
            selected_unannotated = random.sample(unannotated_ids, actual_unannotated) if actual_unannotated > 0 else []
            
            self.image_ids = selected_annotated + selected_unannotated
            print(f"子采样结果：选择 {len(selected_annotated)} 有标注图像 + {len(selected_unannotated)} 无标注图像")
        else:
            # 如果不子采样，优先排列有标注的图像
            self.image_ids = annotated_ids + unannotated_ids
        
        print(f"加载 {split} 数据集，包含 {len(self.image_ids)} 张图像")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取一个数据样本
        
        Returns
        -------
        pixel_data : torch.Tensor with shape (num_pixels, 3)
            像素数据：[R, G, B]
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
        """创建分割掩码 - 改进版本，使用精确的多边形分割"""
        H, W = image_shape
        mask = np.zeros((H, W), dtype=np.int64)
        
        # 获取该图像的所有标注
        annotations = self.image_annotations.get(image_id, [])
        
        for i, ann in enumerate(annotations):
            category_id = ann['category_id']
            
            # 优先使用精确的分割多边形
            if 'segmentation' in ann and ann['segmentation']:
                segmentation = ann['segmentation']
                
                # 处理多边形分割
                if isinstance(segmentation, list) and len(segmentation) > 0:
                    # 多边形格式：[[x1, y1, x2, y2, ...]]
                    polygons = segmentation
                    
                    # 将COCO海岸养殖地物类别(0-5)映射到连续标签(1-6)，保留0作为背景
                    label = category_id + 1
                    
                    # 创建多边形掩码
                    from PIL import Image, ImageDraw
                    img = Image.new('L', (W, H), 0)
                    draw = ImageDraw.Draw(img)
                    
                    for polygon in polygons:
                        # 将坐标缩放到当前图像尺寸
                        orig_w = self.image_info[image_id]['width']
                        orig_h = self.image_info[image_id]['height']
                        
                        scaled_polygon = []
                        for j in range(0, len(polygon), 2):
                            x = int(polygon[j] * W / orig_w)
                            y = int(polygon[j + 1] * H / orig_h)
                            x = max(0, min(x, W-1))
                            y = max(0, min(y, H-1))
                            scaled_polygon.extend([x, y])
                        
                        if len(scaled_polygon) >= 6:  # 至少3个点
                            draw.polygon(scaled_polygon, fill=label)
                    
                    # 转换为numpy数组并合并到主掩码
                    polygon_mask = np.array(img)
                    mask = np.maximum(mask, polygon_mask)
                
                # 处理RLE格式分割
                elif isinstance(segmentation, dict) and 'counts' in segmentation:
                    try:
                        from pycocotools import mask as maskUtils
                        rle = segmentation
                        binary_mask = maskUtils.decode(rle)
                        
                        # 缩放掩码到目标尺寸
                        from PIL import Image
                        pil_mask = Image.fromarray(binary_mask.astype(np.uint8))
                        pil_mask = pil_mask.resize((W, H), Image.NEAREST)
                        binary_mask = np.array(pil_mask)
                        
                        # 将COCO类别映射到标签
                        label = category_id + 1
                        mask[binary_mask > 0] = label
                        
                    except ImportError:
                        print("警告：pycocotools未安装，无法处理RLE格式分割，回退到bbox")
                        # 回退到bbox方法
                        self._apply_bbox_mask(ann, mask, image_id, H, W)
            
            # 如果没有分割信息，使用bbox作为回退
            elif 'bbox' in ann and ann['bbox']:
                self._apply_bbox_mask(ann, mask, image_id, H, W)
        
        return mask
    
    def _apply_bbox_mask(self, ann: Dict, mask: np.ndarray, image_id: int, H: int, W: int):
        """应用边界框掩码（回退方法）"""
        category_id = ann['category_id']
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, w, h = bbox
        
        # 跳过无效的边界框
        if w <= 0 or h <= 0:
            return
        
        # 将边界框坐标缩放到当前图像尺寸
        orig_w = self.image_info[image_id]['width']
        orig_h = self.image_info[image_id]['height']
        
        x = int(x * W / orig_w)
        y = int(y * H / orig_h) 
        w = int(w * W / orig_w)
        h = int(h * H / orig_h)
        
        # 确保坐标在有效范围内
        x = max(0, min(x, W-1))
        y = max(0, min(y, H-1))
        w = max(1, min(w, W-x))
        h = max(1, min(h, H-y))
        
        # 将COCO海岸养殖地物类别(0-5)映射到连续标签(1-6)，保留0作为背景
        label = category_id + 1
        
        mask[y:y+h, x:x+w] = label
    
    def _image_to_pixels(self, image: torch.Tensor, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """将图像转换为像素级数据（仅 RGB 通道）"""
        C, H, W = image.shape
        
        # 重塑数据
        rgb_values = image.permute(1, 2, 0).flatten(0, 1)  # (H*W, 3)
        labels = torch.from_numpy(mask).flatten()  # (H*W,)
        
        # 像素数据：[R, G, B]
        pixel_data = rgb_values  # (H*W, 3)
        
        # 检查是否需要智能采样
        # 对于轴向GATr，我们需要完整的图像像素
        total_pixels = pixel_data.shape[0]
        
        # 如果像素数量超过限制且不是轴向GATr，则进行智能采样
        if total_pixels > self.max_items:
            # 找到前景和背景像素
            fg_mask = labels > 0
            bg_mask = labels == 0
            
            fg_indices = torch.where(fg_mask)[0]
            bg_indices = torch.where(bg_mask)[0]
            
            # 前景像素优先策略：保留70%前景，30%背景
            if len(fg_indices) > 0:
                max_fg = min(int(self.max_items * 0.7), len(fg_indices))
                max_bg = self.max_items - max_fg
                
                # 采样前景像素
                if len(fg_indices) > max_fg:
                    selected_fg = fg_indices[torch.randperm(len(fg_indices))[:max_fg]]
                else:
                    selected_fg = fg_indices
                
                # 采样背景像素
                if len(bg_indices) > max_bg:
                    selected_bg = bg_indices[torch.randperm(len(bg_indices))[:max_bg]]
                else:
                    selected_bg = bg_indices
                
                indices = torch.cat([selected_fg, selected_bg])
            else:
                # 如果没有前景像素，随机采样
                indices = torch.randperm(total_pixels)[:self.max_items]
            
            pixel_data = pixel_data[indices]
            labels = labels[indices]
        
        return pixel_data, labels


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """自定义批处理函数，处理不同数量像素的图像
    
    使用padding方式而不是concatenation，保持批次维度
    """
    pixel_data_list = []
    labels_list = []
    
    for pixel_data, labels in batch:
        pixel_data_list.append(pixel_data)
        labels_list.append(labels)
    
    # 找到批次中最大的像素数量
    max_pixels = max(pixel_data.shape[0] for pixel_data in pixel_data_list)
    
    # 对每个样本进行padding
    padded_pixel_data = []
    padded_labels = []
    
    for pixel_data, labels in zip(pixel_data_list, labels_list):
        current_pixels = pixel_data.shape[0]
        pad_size = max_pixels - current_pixels
        
        if pad_size > 0:
            # 对像素数据进行padding
            padded_pixels = F.pad(pixel_data, (0, 0, 0, pad_size), value=0.0)
            # 对标签进行padding，使用-1表示无效标签
            padded_label = F.pad(labels, (0, pad_size), value=-1)
        else:
            padded_pixels = pixel_data
            padded_label = labels
        
        padded_pixel_data.append(padded_pixels)
        padded_labels.append(padded_label)
    
    # 堆叠为批次
    batch_pixel_data = torch.stack(padded_pixel_data, dim=0)  # (batch_size, max_pixels, 5)
    batch_labels = torch.stack(padded_labels, dim=0)          # (batch_size, max_pixels)
    
    return batch_pixel_data, batch_labels