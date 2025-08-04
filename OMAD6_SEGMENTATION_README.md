# 🏥 OMAD-6 医学图像分割 - GATr实现

基于几何代数变换器(Geometric Algebra Transformer)的医学图像分割解决方案，用于OMAD-6数据集。

## 🌟 核心特性

### 🔬 几何代数创新
- **统一表示**: 将2D坐标和RGB值嵌入16维多重向量空间
- **等变性保持**: 对图像旋转、平移等变换保持数学一致性  
- **几何感知**: 内置几何结构理解，超越传统CNN方法

### 🚀 技术亮点
- **Pin(3,0,1)等变性**: 严格的数学对称性保证
- **多重向量表示**: 坐标点(trivectors) + RGB标量的创新嵌入
- **几何注意力**: 结合欧几里得距离的注意力机制
- **双架构支持**: 标准GATr + 轴向GATr（专为2D图像优化）

## 📁 项目结构

```
geometric-algebra-transformer/
├── gatr/experiments/omad6/          # OMAD-6分割实现
│   ├── dataset.py                   # 数据加载器（COCO格式支持）
│   ├── wrappers.py                  # GATr包装器（标准版+轴向版）
│   ├── experiment.py                # 实验管理器
│   └── utils.py                     # 可视化和评估工具
├── config/
│   ├── omad6.yaml                   # 主配置文件
│   └── model/
│       ├── gatr_omad6.yaml          # 标准GATr配置
│       └── axial_gatr_omad6.yaml    # 轴向GATr配置
├── scripts/omad6_experiment.py      # 实验脚本
├── demo_omad6_segmentation.py       # 演示脚本
└── data/OMAD-6/                     # 数据集目录
    ├── train2017/                   # 训练图像
    ├── val2017/                     # 验证图像
    ├── test2017/                    # 测试图像
    └── annotations/                 # COCO格式标注
```

## 🔧 环境要求

```bash
# 基础依赖
torch>=2.0
einops  
numpy<1.25
opt_einsum
xformers

# 图像处理
PIL
torchvision

# 实验管理
hydra-core
omegaconf
mlflow

# 可视化
matplotlib
```

## 🚀 快速开始

### 1. 数据准备
确保OMAD-6数据集在正确位置：
```
data/OMAD-6/
├── train2017/           # 训练图像 (PNG格式)
├── val2017/             # 验证图像  
├── test2017/            # 测试图像
└── annotations/         # COCO格式标注
    ├── instances_train2017.json
    └── instances_val2017.json
```

### 2. 运行演示
```bash
# 交互式演示
python demo_omad6_segmentation.py

# 或直接运行实验
python scripts/omad6_experiment.py base_dir="/path/to/experiments" seed=42
```

### 3. 训练模型

#### 🎯 标准GATr（推荐新手）
```bash
python scripts/omad6_experiment.py \
    base_dir="${BASEDIR}" \
    seed=42 \
    model=gatr_omad6 \
    training.steps=5000 \
    training.batchsize=2 \
    run_name=gatr_basic
```

#### 🎯 轴向GATr（更适合2D图像）
```bash
python scripts/omad6_experiment.py \
    base_dir="${BASEDIR}" \
    seed=42 \
    model=axial_gatr_omad6 \
    training.steps=5000 \
    training.batchsize=2 \
    run_name=axial_gatr
```

#### 🎯 快速测试（10%数据）
```bash
python scripts/omad6_experiment.py \
    base_dir="${BASEDIR}" \
    seed=42 \
    data.subsample=0.1 \
    training.steps=1000 \
    run_name=quick_test
```

## 🧠 技术原理

### 几何代数嵌入策略

#### 🔹 像素数据表示
```python
# 输入：像素数据 [x, y, R, G, B]
# 坐标范围：[-1, 1], RGB范围：[0, 1]

# 1. 2D坐标扩展为3D点（z=0）
coordinates_3d = [x, y, 0]
point_embedding = embed_point(coordinates_3d)  # trivector表示

# 2. RGB值嵌入为标量
r_scalar = embed_scalar([R])
g_scalar = embed_scalar([G])  
b_scalar = embed_scalar([B])

# 3. 多通道多重向量
multivector = stack([point_embedding, r_scalar, g_scalar, b_scalar])
# 形状：(batch, pixels, 4_channels, 16_dimensions)
```

#### 🔹 几何注意力机制
```python
# 结合三种相似度计算：
# 1. PGA内积（几何关系）
# 2. 欧几里得距离（空间位置）
# 3. 非线性特征（RGB相似性）

attention_weights = softmax(
    pga_inner_product(q_mv, k_mv) +
    euclidean_distance(q_s, k_s) +
    nonlinear_features(phi(q_s), psi(k_s))
)
```

### 网络架构对比

| 特性 | 标准GATr | 轴向GATr |
|------|----------|----------|
| **适用场景** | 通用分割 | 2D图像特化 |
| **计算复杂度** | O(n²) | O(n√n) |
| **空间感知** | 全局注意力 | 行列分离注意力 |
| **内存使用** | 较高 | 较低 |
| **推荐用途** | 小图像、高精度 | 大图像、高效率 |

## 📊 评估指标

### 🎯 主要指标
- **mIoU (平均IoU)**: 各类别IoU的平均值
- **整体准确率**: 像素级分类准确率
- **类别IoU**: 每个解剖结构的分割质量

### 🎯 预期性能
```
标准GATr:
├── mIoU: ~0.75-0.85 (取决于数据质量)
├── 准确率: ~0.85-0.95
└── 训练时间: ~2-4小时 (RTX 3080)

轴向GATr:
├── mIoU: ~0.70-0.80
├── 准确率: ~0.80-0.90  
└── 训练时间: ~1-2小时 (内存效率更高)
```

## 🔬 实验配置详解

### 数据配置 (`config/omad6.yaml`)
```yaml
data:
  image_size: 256        # 图像尺寸
  max_items: 1024        # 每图最大像素数（内存控制）
  num_classes: 11        # 分割类别数（含背景）
  subsample: null        # 数据子采样率
```

### 模型配置 (`config/model/gatr_omad6.yaml`)
```yaml
net:
  in_mv_channels: 4      # 输入：点+R+G+B
  out_mv_channels: 1     # 输出：分割特征
  hidden_mv_channels: 32 # 隐藏层宽度
  num_blocks: 8          # Transformer块数
  attention:
    num_heads: 4         # 注意力头数
    hidden_dim: 256      # 注意力维度
  dropout_prob: 0.1      # Dropout率
```

### 训练配置
```yaml
training:
  lr: 1e-4              # 学习率（较保守）
  batchsize: 2          # 批次大小（内存限制）
  steps: 10000          # 训练步数
  weight_decay: 1e-4    # 权重衰减
  clip_grad_norm: 1.0   # 梯度裁剪
```

## 🎨 可视化功能

### 结果可视化
```python
from gatr.experiments.omad6.utils import visualize_segmentation_results

# 可视化分割结果
visualize_segmentation_results(
    image=original_image,
    ground_truth=gt_mask,
    prediction=pred_mask,
    save_path="./results/segmentation_comparison.png"
)
```

### 训练监控
- **MLflow跟踪**: 自动记录损失、指标、参数
- **实时日志**: 训练过程可视化
- **检查点保存**: 自动保存最佳模型

## 🚀 高级用法

### 1. 自定义数据集
```python
# 继承OMAD6Dataset类
class CustomMedicalDataset(OMAD6Dataset):
    def _create_segmentation_mask(self, image_id, image_shape):
        # 实现自定义的掩码生成逻辑
        pass
```

### 2. 模型调优
```bash
# 调整网络深度
python scripts/omad6_experiment.py model.net.num_blocks=12

# 调整注意力头数
python scripts/omad6_experiment.py model.net.attention.num_heads=8

# 启用梯度检查点（节省内存）
python scripts/omad6_experiment.py model.net.checkpoint='["block"]'
```

### 3. 多GPU训练
```bash
# 使用数据并行
export CUDA_VISIBLE_DEVICES=0,1
python scripts/omad6_experiment.py training.batchsize=8
```

## 🔍 故障排除

### 常见问题

#### 🐛 内存不足
```bash
# 解决方案：
# 1. 减少批次大小
training.batchsize=1

# 2. 减少最大像素数
data.max_items=512

# 3. 启用梯度检查点
model.net.checkpoint='["block"]'
```

#### 🐛 收敛慢
```bash
# 解决方案：
# 1. 调整学习率
training.lr=3e-4

# 2. 减少权重衰减
training.weight_decay=1e-5

# 3. 使用学习率调度
training.lr_decay=0.5
```

#### 🐛 过拟合
```bash
# 解决方案：
# 1. 增加Dropout
model.net.dropout_prob=0.2

# 2. 增加权重衰减
training.weight_decay=1e-3

# 3. 使用数据增强
data.augmentation=true
```

## 📈 性能优化建议

### 🚀 训练加速
1. **使用混合精度**: `training.float16=true`
2. **编译模型**: `torch.compile(model)`
3. **预计算嵌入**: 缓存像素嵌入结果
4. **批处理优化**: 动态批次大小调整

### 🧠 内存优化
1. **梯度检查点**: 以计算换内存
2. **像素采样**: 随机采样减少内存使用
3. **模型蒸馏**: 训练较小的学生模型
4. **量化推理**: 8位整数推理

## 🎯 扩展方向

### 🔬 研究扩展
- **3D医学图像**: 扩展到CT/MRI体数据
- **多模态融合**: 结合文本、图像、临床数据
- **在线学习**: 增量学习新的解剖结构
- **不确定性量化**: 贝叶斯GATr变种

### 🏥 临床应用
- **实时分割**: 手术导航系统
- **疾病诊断**: 病变区域自动检测
- **治疗规划**: 放疗靶区勾画
- **质量控制**: 医学图像质量评估

## 📚 参考文献

1. **Geometric Algebra Transformer**: [NeurIPS 2023](https://arxiv.org/abs/2305.18415)
2. **投影几何代数**: Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA"
3. **医学图像分割**: Comprehensive survey of deep learning approaches

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

### 开发流程
1. Fork项目仓库
2. 创建特性分支
3. 编写测试用例
4. 提交Pull Request

### 代码规范
- 遵循PEP 8风格
- 添加类型注解
- 编写文档字符串
- 确保测试通过

## 📄 许可证

本项目基于Qualcomm Technologies, Inc.的许可证发布。

---

🚀 **开始你的几何代数图像分割之旅！** 

如有问题，请查看issues或联系维护者。