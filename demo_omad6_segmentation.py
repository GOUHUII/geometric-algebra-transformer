#!/usr/bin/env python3
"""
OMAD-6 图像分割演示脚本

这个脚本演示了如何使用GATr进行医学图像分割任务。

使用步骤：
1. 确保OMAD-6数据集在 data/OMAD-6/ 目录下
2. 运行此脚本开始训练
3. 监控训练过程和评估结果

作者：基于Geometric Algebra Transformer项目开发
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """主演示函数"""
    print("🏥 OMAD-6 医学图像分割 - GATr实现")
    print("=" * 50)
    
    # 检查数据集
    data_dir = project_root / "data" / "OMAD-6"
    if not data_dir.exists():
        print(f"❌ 错误：数据集目录不存在: {data_dir}")
        print("请确保OMAD-6数据集已下载到正确位置")
        return
    
    print(f"✅ 数据集目录找到: {data_dir}")
    
    # 创建实验目录
    exp_dir = project_root / "experiments"
    exp_dir.mkdir(exist_ok=True)
    
    print("\n🚀 可用的训练命令：")
    print()
    
    # 基础GATr训练
    print("1. 基础GATr模型训练：")
    cmd1 = f"""python scripts/omad6_experiment.py \\
    base_dir="{exp_dir}" \\
    seed=42 \\
    model=gatr_omad6 \\
    training.steps=5000 \\
    training.batchsize=2 \\
    run_name=gatr_basic"""
    print(f"   {cmd1}")
    print()
    
    # 轴向GATr训练
    print("2. 轴向GATr模型训练（更适合2D图像）：")
    cmd2 = f"""python scripts/omad6_experiment.py \\
    base_dir="{exp_dir}" \\
    seed=42 \\
    model=axial_gatr_omad6 \\
    training.steps=5000 \\
    training.batchsize=2 \\
    run_name=axial_gatr"""
    print(f"   {cmd2}")
    print()
    
    # 快速测试
    print("3. 快速测试（使用10%数据）：")
    cmd3 = f"""python scripts/omad6_experiment.py \\
    base_dir="{exp_dir}" \\
    seed=42 \\
    model=gatr_omad6 \\
    data.subsample=0.1 \\
    training.steps=1000 \\
    training.batchsize=1 \\
    run_name=quick_test"""
    print(f"   {cmd3}")
    print()
    
    # 交互式选择
    print("🎯 选择要运行的命令（输入1-3）：")
    try:
        choice = input("请输入选择（或按Enter跳过）: ").strip()
        
        if choice == "1":
            print("\n🔥 开始基础GATr训练...")
            os.system(cmd1)
        elif choice == "2":
            print("\n🔥 开始轴向GATr训练...")
            os.system(cmd2)
        elif choice == "3":
            print("\n🔥 开始快速测试...")
            os.system(cmd3)
        else:
            print("\n💡 提示：复制上述命令到终端中运行")
            
    except KeyboardInterrupt:
        print("\n👋 演示结束")
    
    print("\n📊 训练完成后，检查以下目录：")
    print(f"   - 实验结果: {exp_dir}")
    print(f"   - MLflow跟踪: {exp_dir}/tracking/")
    print()
    print("📈 主要评估指标：")
    print("   - mean_iou: 平均IoU（越高越好）")
    print("   - overall_accuracy: 整体准确率")
    print("   - loss: 训练损失（越低越好）")
    print()
    print("🔬 技术特性：")
    print("   ✨ 几何代数表示：将2D坐标和RGB值嵌入16维多重向量")
    print("   ✨ 等变性保持：对旋转、平移等变换保持一致性")
    print("   ✨ 统一架构：单一网络处理多种几何数据类型")
    print("   ✨ 可扩展性：支持大规模医学图像分割")


if __name__ == "__main__":
    main()