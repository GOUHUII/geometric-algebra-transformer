#!/usr/bin/env python3
# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
"""OMAD-6 海岸养殖地物遥感图像分割实验脚本

使用GATr进行海岸养殖地物遥感图像分割的训练和评估脚本。

用法:
    python scripts/omad6_experiment.py base_dir="/path/to/experiments" seed=42

示例:
    # 使用GATr训练分割模型
    python scripts/omad6_experiment.py base_dir="/tmp/gatr-experiments" \\
        seed=42 model=gatr_omad6 training.steps=5000 run_name=gatr_seg
    
    # 使用轴向GATr训练
    python scripts/omad6_experiment.py base_dir="/tmp/gatr-experiments" \\
        seed=42 model=axial_gatr_omad6 training.steps=5000 run_name=axial_gatr_seg
    
    # 快速实验（使用数据子集）
    python scripts/omad6_experiment.py base_dir="/tmp/gatr-experiments" \\
        seed=42 data.subsample=0.1 training.steps=1000 run_name=quick_test
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gatr.experiments.omad6.experiment import OMAD6Experiment

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_data_directory(data_dir: Path) -> bool:
    """验证数据目录的完整性
    
    Parameters
    ----------
    data_dir : Path
        数据目录路径
        
    Returns
    -------
    bool
        数据目录是否有效
    """
    logger.info(f"验证数据目录: {data_dir}")
    
    # 检查主目录
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return False
    
    # 检查图像目录
    required_splits = ["train2017", "val2017", "test2017"]
    for split in required_splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            logger.warning(f"图像目录不存在: {split_dir}")
        else:
            # 检查是否有图像文件
            image_files = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
            logger.info(f"{split} 目录包含 {len(image_files)} 张图像")
    
    # 检查标注目录
    annotations_dir = data_dir / "annotations"
    if not annotations_dir.exists():
        logger.error(f"标注目录不存在: {annotations_dir}")
        return False
    
    # 检查标注文件
    required_annotations = ["instances_train2017.json", "instances_val2017.json"]
    for ann_file in required_annotations:
        ann_path = annotations_dir / ann_file
        if not ann_path.exists():
            logger.error(f"标注文件不存在: {ann_path}")
            return False
        else:
            logger.info(f"找到标注文件: {ann_path}")
    
    # 检查测试标注（可选）
    test_ann_path = annotations_dir / "instances_test2017.json"
    if test_ann_path.exists():
        logger.info(f"找到测试标注文件: {test_ann_path}")
    else:
        logger.warning(f"测试标注文件不存在: {test_ann_path}")
    
    logger.info("数据目录验证完成")
    return True


@hydra.main(version_base=None, config_path="../config", config_name="omad6")
def main(cfg: DictConfig) -> None:
    """主函数：运行OMAD-6分割实验
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra配置对象
    """
    logger.info("开始OMAD-6海岸养殖地物遥感图像分割实验")
    logger.info(f"实验配置: {cfg}")
    
    # 验证数据路径
    data_dir = Path(cfg.data.data_dir)
    if not validate_data_directory(data_dir):
        raise FileNotFoundError(f"数据目录验证失败: {data_dir}")
    
    try:
        # 创建实验实例
        experiment = OMAD6Experiment(cfg)
        
        # 运行实验
        logger.info("开始训练...")
        experiment()
        
        logger.info("实验完成！")
        
    except Exception as e:
        logger.error(f"实验运行失败: {e}")
        raise


if __name__ == "__main__":
    main()