#!/usr/bin/env python3
# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
"""OMAD-6 医学图像分割实验脚本

使用GATr进行医学图像分割的训练和评估脚本。

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


@hydra.main(version_base=None, config_path="../config", config_name="omad6")
def main(cfg: DictConfig) -> None:
    """主函数：运行OMAD-6分割实验
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra配置对象
    """
    logger.info("开始OMAD-6图像分割实验")
    logger.info(f"实验配置: {cfg}")
    
    # 验证数据路径
    data_dir = Path(cfg.data.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    # 检查必要的数据文件
    train_images = data_dir / "train2017"
    train_annotations = data_dir / "annotations" / "instances_train2017.json"
    
    if not train_images.exists():
        raise FileNotFoundError(f"训练图像目录不存在: {train_images}")
    if not train_annotations.exists():
        raise FileNotFoundError(f"训练标注文件不存在: {train_annotations}")
    
    logger.info(f"数据目录验证通过: {data_dir}")
    
    try:
        # 创建实验实例
        experiment = OMAD6Experiment(cfg)
        
        # 运行实验
        logger.info("开始训练...")
        experiment.run()
        
        logger.info("实验完成！")
        
    except Exception as e:
        logger.error(f"实验运行失败: {e}")
        raise


if __name__ == "__main__":
    main()