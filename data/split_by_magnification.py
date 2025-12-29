"""
按单一放大倍数划分数据集
确保同一病人的所有图像都在同一个集合中（防止数据泄露）
"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os
import glob
import argparse


def split_by_magnification(
    dataset_root, 
    magnification,
    output_dir="./", 
    test_size=0.15, 
    val_size=0.176, 
    random_state=42
):
    """
    根据指定放大倍数划分数据集为训练集、验证集和测试集
    确保同一病人的所有图像都在同一个集合中
    
    Args:
        dataset_root: 数据集根目录
        magnification: 放大倍数 (40, 100, 200, 400)
        output_dir: 输出CSV文件的目录
        test_size: 测试集比例
        val_size: 验证集比例（相对于train+val的比例）
        random_state: 随机种子
    
    Returns:
        train_df, val_df, test_df: 三个DataFrame
    """
    print(f"正在从 {dataset_root} 读取图片...")
    print(f"目标放大倍数: {magnification}X")
    
    # 1. 读取所有图片路径
    all_image_paths = glob.glob(os.path.join(dataset_root, "**", "*.png"), recursive=True)
    
    if len(all_image_paths) == 0:
        raise ValueError(f"在 {dataset_root} 中未找到任何PNG图片！")
    
    data = []
    for path in all_image_paths:
        filename = os.path.basename(path)
        # 解析文件名，提取关键信息
        # 示例文件名: SOB_B_TA-14-4659-40-001.png
        parts = filename.split('-')
        
        # 提取放大倍数
        mag = parts[3] if len(parts) > 3 else "unknown"
        
        # 只处理指定放大倍数的图片
        if mag != str(magnification):
            continue
        
        # 提取病人ID (Patient ID)
        # 根据BreaKHis格式，病人ID是parts[2]，如 '21998EF', '15572'
        if len(parts) >= 3:
            patient_id = parts[2]
        else:
            # 如果文件名格式不符合预期，使用文件名作为patient_id
            patient_id = filename.split('.')[0]
        
        # 提取类别 (Class) - 假设父文件夹的名字就是类别
        label = os.path.basename(os.path.dirname(path))
        
        # 二分类标签 (Binary)
        binary_label = "Benign" if "B_" in filename else "Malignant"
        
        # 提取magnification_folder（从路径中提取，如 40X, 100X等）
        path_parts = path.split(os.sep)
        magnification_folder = path_parts[-2] if len(path_parts) >= 2 else f"{magnification}X"
        
        data.append({
            "path": path,
            "filename": filename,
            "patient_id": patient_id,
            "class": label,
            "binary_label": binary_label,
            "magnification": mag,
            "magnification_folder": magnification_folder
        })
    
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        raise ValueError(f"未找到放大倍数为 {magnification}X 的图片！")
    
    print(f"\n找到 {len(df)} 张 {magnification}X 的图片")
    print(f"总病人数: {df['patient_id'].nunique()}")
    print(f"类别分布:\n{df['class'].value_counts()}")
    print(f"二分类标签分布:\n{df['binary_label'].value_counts()}")
    
    # ---------------------------------------------------------
    # 2. 开始划分 (Train / Val / Test)
    # 使用 GroupShuffleSplit 保证同一个 patient_id 不会跨越数据集
    # ---------------------------------------------------------
    
    # 第一刀：切分出 Test Set
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(splitter_test.split(df, groups=df['patient_id']))
    
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    # 第二刀：从剩下的里面切分出 Validation Set
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(splitter_val.split(train_val_df, groups=train_val_df['patient_id']))
    
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    
    # ---------------------------------------------------------
    # 3. 验证分布 (Sanity Check)
    # ---------------------------------------------------------
    print_stats("Train", train_df)
    print_stats("Validation", val_df)
    print_stats("Test", test_df)
    
    # 检查是否有病人ID重叠
    train_patients = set(train_df['patient_id'].unique())
    val_patients = set(val_df['patient_id'].unique())
    test_patients = set(test_df['patient_id'].unique())
    
    assert len(train_patients & val_patients) == 0, "训练集和验证集有病人ID重叠！"
    assert len(train_patients & test_patients) == 0, "训练集和测试集有病人ID重叠！"
    assert len(val_patients & test_patients) == 0, "验证集和测试集有病人ID重叠！"
    print("\n✓ 数据划分验证通过：各数据集病人ID无重叠")
    
    # 检查所有图片都是指定放大倍数
    assert all(train_df['magnification'] == str(magnification)), "训练集中存在非目标放大倍数的图片！"
    assert all(val_df['magnification'] == str(magnification)), "验证集中存在非目标放大倍数的图片！"
    assert all(test_df['magnification'] == str(magnification)), "测试集中存在非目标放大倍数的图片！"
    print(f"✓ 所有数据集都只包含 {magnification}X 的图片")
    
    # 4. 保存为CSV
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, f"train_split_{magnification}x.csv")
    val_path = os.path.join(output_dir, f"val_split_{magnification}x.csv")
    test_path = os.path.join(output_dir, f"test_split_{magnification}x.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✓ 数据集划分完成！CSV文件已保存到 {output_dir}")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")
    
    return train_df, val_df, test_df


def print_stats(name, dataframe):
    """打印数据集统计信息"""
    print(f"\n--- {name} Set ---")
    print(f"Images: {len(dataframe)}")
    print(f"Patients: {dataframe['patient_id'].nunique()}")
    print("Class Distribution:")
    print(dataframe['class'].value_counts())
    print("Binary Label Distribution:")
    print(dataframe['binary_label'].value_counts(normalize=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按单一放大倍数划分数据集")
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='BreaKHis_v1/BreaKHis_v1/histology_slides/breast',
        help='数据集根目录'
    )
    parser.add_argument(
        '--magnification',
        type=int,
        required=True,
        choices=[40, 100, 200, 400],
        help='放大倍数 (40, 100, 200, 400)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./',
        help='输出CSV文件的目录'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.15,
        help='测试集比例'
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.176,
        help='验证集比例（相对于train+val的比例）'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    split_by_magnification(
        dataset_root=args.dataset_root,
        magnification=args.magnification,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )

