# Cancer Classification with Multi-Task Learning & DAF-MMD-Net

本项目专注于基于 **BreakHis (Breast Cancer Histopathological Database)** 数据集的乳腺癌病理图像分类研究。

主要实现的模型包括：
1.  **DAF-MMD-Net (Xception-Xception)**: 使用双流 Xception 网络结合 MMD (Maximum Mean Discrepancy) 损失，同时利用 40x 和 100x 放大倍率的图像特征。
2.  **Multi-Task Learning**: 同时进行二分类 (良性/恶性) 和八分类 (具体亚型) 任务。

## 📂 数据集下载

本项目的原始数据集太大，无法直接上传到 GitHub。您可以从 Kaggle 下载：
[BreakHis - Breast Cancer Histopathological Dataset](https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset)

下载后请解压并按照配置文件中的路径进行放置（默认为 `dataset_cancer_v1/`）。

## 🚀 环境配置与激活

本项目使用 Python 虚拟环境来管理依赖。请按照以下步骤激活环境：

```bash
# 激活名为 venv_cancer 的虚拟环境
source venv_cancer/bin/activate
```

*如果这是您第一次运行，请确保已安装 `requirements.txt` 中的依赖（假设环境已存在，此步可选）：*
```bash
pip install -r requirements.txt
```

## 🛠️ 如何运行

### 1. 训练模型 (Training)
主要使用 `train_daf_mmd_xception.py` 脚本进行训练。

**示例：训练 DAF-MMD-Net (Xception-Xception)**
```bash
python train_daf_mmd_xception.py --config configs/daf_mmd_xception_xception.yaml
```

**其他可选配置：**
- Xception-Swin 版本: `configs/daf_mmd_xception_swin.yaml`
- 单倍率/多任务基线: 使用 `train_multitask.py` 配合 `configs/multitask_*.yaml`

### 2. 评估模型 (Evaluation)
训练完成后，使用评估脚本测试模型性能。

**示例：评估 DAF-MMD-Net**
```bash
# 需要指定 checkpoint 目录和配置文件
python evaluate_daf_mmd_xception.py \
    --checkpoint_dir multitask_results/checkpoints/daf_mmd_xception_xception \
    --config configs/daf_mmd_xception_xception.yaml
```

## 📊 实验结果

所有的实验结果（包括日志、混淆矩阵图片、指标 JSON 文件）都保存在 `multitask_results/results/` 目录下。

以 **DAF-MMD (Xception-Xception)** 为例，结果位于：
`multitask_results/results/daf_mmd_xception_xception/`

该目录下包含：
- **mixed/**: 综合 40x 和 100x 样本的总体评估结果。
- **40x/**: 仅基于 40x 图像的评估结果。
- **100x/**: 仅基于 100x 图像的评估结果。
- `training_history.png`: 训练过程中的 Loss 和 Accuracy 曲线。
- `confusion_matrix_*.png`: 详细的混淆矩阵可视化。

## 📝 核心文件说明

- `models/daf_mmd_net_xception.py`: DAF-MMD 网络结构定义。
- `train_daf_mmd_xception.py`: DAF-MMD 训练主程序。
- `evaluate_daf_mmd_xception.py`: DAF-MMD 评估程序。
- `configs/`: 存放所有实验的 YAML 配置文件。
