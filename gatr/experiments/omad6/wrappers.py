# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
"""OMAD-6 图像分割的 GATr 包装器"""

import torch
import torch.nn.functional as F
from typing import Tuple

from gatr.experiments.base_wrapper import BaseWrapper
from gatr.interface import embed_point, embed_scalar, extract_scalar


def embed_pixel_data_in_pga(pixel_data: torch.Tensor) -> torch.Tensor:
    """将像素数据嵌入到投影几何代数中
    
    将2D像素坐标和RGB值嵌入为多重向量。坐标被嵌入为点(trivectors)，
    RGB值被嵌入为标量。
    
    Parameters
    ----------
    pixel_data : torch.Tensor with shape (batch_size, num_pixels, 5)
        像素数据: [x坐标, y坐标, R, G, B]，坐标范围为[-1, 1]，RGB范围为[0, 1]
    
    Returns
    -------
    multivector : torch.Tensor with shape (batch_size, num_pixels, 4, 16)
        几何代数嵌入，包含4个通道：
        - 通道0: 2D坐标点 (使用z=0扩展为3D)
        - 通道1: R通道标量
        - 通道2: G通道标量  
        - 通道3: B通道标量
    """
    batch_size, num_pixels, _ = pixel_data.shape
    
    # 提取坐标和RGB值
    coordinates_2d = pixel_data[:, :, :2]  # (batch_size, num_pixels, 2)
    rgb_values = pixel_data[:, :, 2:5]     # (batch_size, num_pixels, 3)
    
    # 将2D坐标扩展为3D（z=0）
    coordinates_3d = torch.cat([
        coordinates_2d, 
        torch.zeros_like(coordinates_2d[:, :, :1])
    ], dim=-1)  # (batch_size, num_pixels, 3)
    
    # 嵌入坐标为点 (trivectors)
    points = embed_point(coordinates_3d)  # (batch_size, num_pixels, 16)
    
    # 嵌入RGB值为标量
    r_scalars = embed_scalar(rgb_values[:, :, [0]])  # (batch_size, num_pixels, 16)
    g_scalars = embed_scalar(rgb_values[:, :, [1]])  # (batch_size, num_pixels, 16)
    b_scalars = embed_scalar(rgb_values[:, :, [2]])  # (batch_size, num_pixels, 16)
    
    # 堆叠为多通道多重向量
    multivector = torch.stack([points, r_scalars, g_scalars, b_scalars], dim=2)
    # (batch_size, num_pixels, 4, 16)
    
    return multivector


class OMAD6GATrWrapper(BaseWrapper):
    """用于OMAD-6图像分割的GATr包装器
    
    这个包装器处理图像像素数据的输入，将其嵌入到几何代数表示中，
    并从输出中提取像素级的分割预测。
    
    Parameters
    ----------
    net : torch.nn.Module
        GATr模型，应该接受4个多重向量通道和1个标量通道的输入，
        返回1个多重向量通道和1个标量通道的输出
    num_classes : int, optional
        分割类别数量，默认为7（包含背景类）
    """
    
    def __init__(self, net: torch.nn.Module, num_classes: int = 7):
        super().__init__(net, scalars=True, return_other=False)
        self.num_classes = num_classes
        self.supports_variable_items = True
        
        # 改进的分类头：使用更多特征
        # 从multivector提取多个特征 + scalar特征
        mv_feature_dim = 16  # multivector的维度
        scalar_feature_dim = 1  # scalar的维度
        total_features = mv_feature_dim + scalar_feature_dim
        
        # 使用多层分类头提升表达能力
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(total_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, num_classes)
        )
        
        # 使用适当的初始化
        for layer in self.classification_head:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    def embed_into_ga(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将原始像素数据嵌入到几何代数表示中
        
        Parameters
        ----------
        inputs : torch.Tensor with shape (batch_size, num_pixels, 5)
            像素数据: [x坐标, y坐标, R, G, B]
        
        Returns
        -------
        mv_inputs : torch.Tensor with shape (batch_size, num_pixels, 4, 16)
            多重向量输入
        scalar_inputs : torch.Tensor with shape (batch_size, num_pixels, 1)
            标量输入（dummy标量）
        """
        # 将像素数据嵌入到PGA中
        mv_inputs = embed_pixel_data_in_pga(inputs)
        
        # 创建dummy标量输入
        batch_size, num_pixels = inputs.shape[:2]
        scalar_inputs = torch.zeros(
            batch_size, num_pixels, 1, 
            device=inputs.device, dtype=inputs.dtype
        )
        
        return mv_inputs, scalar_inputs
    
    def extract_from_ga(self, multivector: torch.Tensor, scalars: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """从几何代数表示中提取分割预测
        
        Parameters
        ----------
        multivector : torch.Tensor with shape (batch_size, num_pixels, 1, 16)
            输出多重向量
        scalars : torch.Tensor with shape (batch_size, num_pixels, 1)
            输出标量
        
        Returns
        -------
        segmentation_logits : torch.Tensor with shape (batch_size, num_pixels, num_classes)
            分割logits
        regularization : None
            正则化项（在此任务中不使用）
        """
        batch_size, num_pixels = multivector.shape[:2]
        
        # 从多重向量中提取完整特征（使用所有16个分量）
        mv_features = multivector.squeeze(-2)  # (batch_size, num_pixels, 16)
        
        # 合并multivector特征和scalar特征
        combined_features = torch.cat([mv_features, scalars], dim=-1)  # (batch_size, num_pixels, 17)
        
        # 确保分类头在正确的设备上
        if self.classification_head[0].weight.device != multivector.device:
            self.classification_head = self.classification_head.to(multivector.device)
        
        # 生成分割logits
        segmentation_logits = self.classification_head(combined_features)  # (batch_size, num_pixels, num_classes)
        
        return segmentation_logits, None


class OMAD6AxialGATrWrapper(BaseWrapper):
    """用于OMAD-6图像分割的轴向GATr包装器
    
    这个版本使用AxialGATr来更好地处理2D图像结构，
    利用图像的空间局部性。
    
    Parameters
    ----------
    net : torch.nn.Module
        AxialGATr模型
    num_classes : int, optional
        分割类别数量，默认为7
    image_size : int, optional
        图像尺寸，用于重塑为2D网格，默认为256
    """
    
    def __init__(self, net: torch.nn.Module, num_classes: int = 7, image_size: int = 128):
        super().__init__(net, scalars=True, return_other=False)
        self.num_classes = num_classes
        self.image_size = image_size
        self.supports_variable_items = False  # 轴向GATr需要固定的2D网格
        
        # 效果优化的分类头：平衡特征维度和内存使用
        mv_feature_dim = 16  # multivector的维度
        scalar_feature_dim = 1  # scalar的维度
        total_features = mv_feature_dim + scalar_feature_dim
        
        # 使用渐进式分类头提升效果
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(total_features, 64),   # 增加第一层容量
            torch.nn.LayerNorm(64),                # 使用LayerNorm替代BatchNorm处理3D输入
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),                # 稍微增加dropout
            torch.nn.Linear(64, 32),               # 渐进式降维
            torch.nn.LayerNorm(32),                # LayerNorm更适合变长序列
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, num_classes)
        )
        
        # 使用适当的初始化
        for layer in self.classification_head:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    def embed_into_ga(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将像素数据嵌入并重塑为2D网格格式
        
        Parameters
        ----------
        inputs : torch.Tensor with shape (batch_size, num_pixels, 5)
            像素数据: [x坐标, y坐标, R, G, B]
        
        Returns
        -------
        mv_inputs : torch.Tensor with shape (batch_size, H, W, 4, 16)
            2D网格多重向量输入
        scalar_inputs : torch.Tensor with shape (batch_size, H, W, 1)
            2D网格标量输入
        """
        batch_size = inputs.shape[0]
        H = W = self.image_size
        
        # 首先进行常规的GA嵌入
        mv_flat = embed_pixel_data_in_pga(inputs)  # (batch_size, num_pixels, 4, 16)
        
        # 检查像素数量是否匹配预期的图像尺寸
        expected_pixels = H * W
        actual_pixels = mv_flat.shape[1]
        
        if actual_pixels != expected_pixels:
            # 如果像素数量不匹配，我们需要调整
            if actual_pixels < expected_pixels:
                # 像素数量不足，进行padding
                pad_size = expected_pixels - actual_pixels
                mv_flat = torch.cat([
                    mv_flat,
                    torch.zeros(batch_size, pad_size, 4, 16, 
                              device=mv_flat.device, dtype=mv_flat.dtype)
                ], dim=1)
            else:
                # 像素数量过多，进行截断
                mv_flat = mv_flat[:, :expected_pixels, :, :]
        
        # 重塑为2D网格 (batch_size, H, W, 4, 16)
        mv_inputs = mv_flat.view(batch_size, H, W, 4, 16)
        
        # 创建2D网格标量输入
        scalar_inputs = torch.zeros(
            batch_size, H, W, 1,
            device=inputs.device, dtype=inputs.dtype
        )
        
        return mv_inputs, scalar_inputs
    
    def extract_from_ga(self, multivector: torch.Tensor, scalars: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """从2D网格格式的几何代数表示中提取分割预测
        
        Parameters
        ----------
        multivector : torch.Tensor with shape (batch_size, H, W, 1, 16)
            输出多重向量网格
        scalars : torch.Tensor with shape (batch_size, H, W, 1)
            输出标量网格
        
        Returns
        -------
        segmentation_logits : torch.Tensor with shape (batch_size, H*W, num_classes)
            分割logits（重塑为像素序列）
        regularization : None
            正则化项
        """
        batch_size, H, W = multivector.shape[:3]
        
        # 提取完整的multivector特征
        mv_features = multivector.squeeze(-2)  # (batch_size, H, W, 16)
        
        # 合并特征
        combined_features = torch.cat([mv_features, scalars], dim=-1)  # (batch_size, H, W, 17)
        
        # 确保分类头在正确的设备上
        if self.classification_head[0].weight.device != multivector.device:
            self.classification_head = self.classification_head.to(multivector.device)
        
        # 生成分割logits
        segmentation_logits = self.classification_head(combined_features)  # (batch_size, H, W, num_classes)
        
        # 重塑为像素序列格式以保持一致性
        segmentation_logits = segmentation_logits.view(batch_size, H*W, self.num_classes)
        
        return segmentation_logits, None