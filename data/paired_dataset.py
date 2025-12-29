"""
配对数据集加载器
支持40X和100X图像的配对加载（用于DAF-MMD Net）
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os


class PairedMagnificationDataset(Dataset):
    """
    配对放大倍率数据集
    同时加载同一患者的40X和100X图像
    
    注意：由于40X和100X图像是非对齐的（非像素对应），
    我们通过patient_id和class进行配对
    """
    def __init__(self, csv_40x, csv_100x, transform_40x=None, transform_100x=None):
        """
        Args:
            csv_40x: 40X图像的CSV文件路径
            csv_100x: 100X图像的CSV文件路径
            transform_40x: 40X图像的变换
            transform_100x: 100X图像的变换
        """
        self.df_40x = pd.read_csv(csv_40x)
        self.df_100x = pd.read_csv(csv_100x)
        
        self.transform_40x = transform_40x
        self.transform_100x = transform_100x
        
        # 创建配对索引
        # 对于每个40X样本，找到对应的100X样本（基于patient_id和class）
        self._create_pairs()
        
        print(f"数据集配对完成:")
        print(f"  40X图像数: {len(self.df_40x)}")
        print(f"  100X图像数: {len(self.df_100x)}")
        print(f"  配对样本数: {len(self.pairs)}")
    
    def _create_pairs(self):
        """
        创建40X和100X图像的配对
        配对策略：基于patient_id和class进行配对
        """
        self.pairs = []
        
        # 为100X数据创建索引（基于patient_id和class）
        df_100x_indexed = {}
        for idx, row in self.df_100x.iterrows():
            key = (row['patient_id'], row['class'])
            if key not in df_100x_indexed:
                df_100x_indexed[key] = []
            df_100x_indexed[key].append(idx)
        
        # 遍历40X数据，找到对应的100X样本
        for idx_40x, row_40x in self.df_40x.iterrows():
            key = (row_40x['patient_id'], row_40x['class'])
            
            if key in df_100x_indexed:
                # 找到匹配的100X样本，随机选择一个（如果有多个）
                import random
                idx_100x = random.choice(df_100x_indexed[key])
                self.pairs.append((idx_40x, idx_100x))
        
        if len(self.pairs) == 0:
            raise ValueError("未找到任何配对的40X和100X样本！请检查CSV文件。")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        返回配对的40X和100X图像及其标签
        
        Returns:
            image_40x: 40X图像
            image_100x: 100X图像
            label_binary: 二分类标签（0=Benign, 1=Malignant）
            label_multiclass: 八分类标签（0-7）
            patient_id: 患者ID
        """
        idx_40x, idx_100x = self.pairs[idx]
        
        # 加载40X图像
        row_40x = self.df_40x.iloc[idx_40x]
        img_path_40x = row_40x['path']
        
        try:
            image_40x = Image.open(img_path_40x).convert('RGB')
        except Exception as e:
            print(f"Error loading 40X image {img_path_40x}: {e}")
            image_40x = Image.new('RGB', (224, 224), color='black')
        
        # 加载100X图像
        row_100x = self.df_100x.iloc[idx_100x]
        img_path_100x = row_100x['path']
        
        try:
            image_100x = Image.open(img_path_100x).convert('RGB')
        except Exception as e:
            print(f"Error loading 100X image {img_path_100x}: {e}")
            image_100x = Image.new('RGB', (224, 224), color='black')
        
        # 应用变换
        if self.transform_40x:
            image_40x = self.transform_40x(image_40x)
        if self.transform_100x:
            image_100x = self.transform_100x(image_100x)
        
        # 获取二分类标签（Benign=0, Malignant=1）
        binary_label = row_40x['binary_label']  # 'Benign' or 'Malignant'
        label_binary = 0 if binary_label == 'Benign' else 1
        
        # 获取八分类标签
        class_name = row_40x['class']
        if not hasattr(self, 'class_to_idx_8'):
            # 定义8个类别的映射
            self.class_names_8 = sorted(['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma',
                                        'ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma'])
            self.class_to_idx_8 = {name: idx for idx, name in enumerate(self.class_names_8)}
        
        label_multiclass = self.class_to_idx_8.get(class_name, 0)  # 默认为0如果找不到
        
        # 患者ID
        patient_id = row_40x['patient_id']
        
        return image_40x, image_100x, label_binary, label_multiclass, patient_id
    
    def get_class_weights(self):
        """
        计算类别权重，用于处理类别不平衡
        """
        labels = []
        for idx_40x, _ in self.pairs:
            row = self.df_40x.iloc[idx_40x]
            binary_label = row['binary_label']
            label = 0 if binary_label == 'Benign' else 1
            labels.append(label)
        
        from collections import Counter
        class_counts = Counter(labels)
        total = len(labels)
        
        weights = torch.tensor([
            1.0 / class_counts.get(0, 1),
            1.0 / class_counts.get(1, 1)
        ], dtype=torch.float)
        weights = weights / weights.sum() * len(class_counts)
        
        return weights


if __name__ == "__main__":
    # 测试代码
    from data.dataset import get_transforms
    
    print("="*60)
    print("测试 PairedMagnificationDataset")
    print("="*60)
    
    # 创建数据集
    transform_40x = get_transforms(image_size=224, is_training=True)
    transform_100x = get_transforms(image_size=224, is_training=True)
    
    dataset = PairedMagnificationDataset(
        csv_40x="train_split_40x.csv",
        csv_100x="train_split_100x.csv",
        transform_40x=transform_40x,
        transform_100x=transform_100x
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试加载一个样本
    img_40x, img_100x, label_binary, label_multiclass, patient_id = dataset[0]
    print(f"\n样本0:")
    print(f"  40X图像形状: {img_40x.shape}")
    print(f"  100X图像形状: {img_100x.shape}")
    print(f"  二分类标签: {label_binary} ({'Benign' if label_binary == 0 else 'Malignant'})")
    print(f"  八分类标签: {label_multiclass}")
    print(f"  患者ID: {patient_id}")
    
    # 测试类别权重
    weights = dataset.get_class_weights()
    print(f"\n类别权重: {weights}")
    
    print("\n✓ PairedMagnificationDataset 测试通过！")

