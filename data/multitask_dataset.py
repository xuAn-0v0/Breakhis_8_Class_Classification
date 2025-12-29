"""
多任务学习数据集
同时提供二分类（良性/恶性）和八分类标签
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import sys

# 添加ai4bio-project-Breakhis到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai4bio-project-Breakhis'))


class MultiTaskDataset(Dataset):
    """
    多任务数据集：同时支持二分类和八分类
    """
    def __init__(self, csv_file, root_dir=None, transform=None):
        """
        Args:
            csv_file: CSV文件路径
            root_dir: 数据根目录（如果CSV中是相对路径）
            transform: 图像变换
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # 八分类类别
        self.class_names_multiclass = sorted(self.df['class'].unique())
        self.multiclass_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names_multiclass)}
        
        # 二分类映射（良性 vs 恶性）
        # 适应不同的CSV格式，有些是缩写(A, DC...)，有些是全称(adenosis, ductal_carcinoma...)
        self.binary_mapping = {}
        benign_names = ['A', 'F', 'PT', 'TA', 'adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma']
        for cls_name in self.class_names_multiclass:
            if cls_name in benign_names:
                self.binary_mapping[cls_name] = 0  # Benign
            else:
                self.binary_mapping[cls_name] = 1  # Malignant
        
        print(f"数据集包含 {len(self.df)} 张图片")
        print(f"八分类类别数: {len(self.class_names_multiclass)}")
        print(f"八分类类别: {self.class_names_multiclass}")
        print(f"二分类映射: {self.binary_mapping}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        返回: (image, label_binary, label_multiclass)
        """
        row = self.df.iloc[idx]
        img_path = row['path']
        
        # 处理路径
        if self.root_dir is not None and not os.path.isabs(img_path):
            img_path = os.path.join(self.root_dir, img_path)
        
        # 八分类标签
        class_name = row['class']
        label_multiclass = self.multiclass_to_idx[class_name]
        
        # 二分类标签
        label_binary = self.binary_mapping[class_name]
        
        # 加载图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个黑色图片作为fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label_binary, label_multiclass


if __name__ == "__main__":
    # 测试数据集
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = MultiTaskDataset(
        csv_file="../ai4bio-project-Breakhis/test_split_40x.csv",
        root_dir="/songjian/zixuan/dataset_cancer_v1",
        transform=transform
    )
    
    print(f"\n测试加载第一个样本:")
    image, label_binary, label_multiclass = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Binary label: {label_binary}")
    print(f"Multiclass label: {label_multiclass}")
