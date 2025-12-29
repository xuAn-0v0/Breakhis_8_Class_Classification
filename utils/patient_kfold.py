"""
患者级K-Fold交叉验证
确保同一患者的所有图像（40X和100X）都在同一个fold中
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
import os


def patient_level_kfold_split(
    csv_40x,
    csv_100x,
    n_splits=5,
    random_state=42,
    output_dir="./"
):
    """
    基于患者ID进行K-Fold交叉验证划分
    
    Args:
        csv_40x: 40X图像的CSV文件路径
        csv_100x: 100X图像的CSV文件路径
        n_splits: K-Fold的折数（默认5）
        random_state: 随机种子
        output_dir: 输出CSV文件的目录
    
    Returns:
        folds: List of dicts, 每个dict包含train/val的40X和100X CSV路径
    """
    # 读取CSV文件
    df_40x = pd.read_csv(csv_40x)
    df_100x = pd.read_csv(csv_100x)
    
    print(f"="*60)
    print(f"患者级K-Fold交叉验证划分 (K={n_splits})")
    print(f"="*60)
    print(f"40X图像数: {len(df_40x)}")
    print(f"100X图像数: {len(df_100x)}")
    
    # 获取所有唯一的患者ID（从40X和100X数据中）
    all_patients_40x = set(df_40x['patient_id'].unique())
    all_patients_100x = set(df_100x['patient_id'].unique())
    all_patients = sorted(list(all_patients_40x.union(all_patients_100x)))
    
    print(f"总患者数: {len(all_patients)}")
    
    # 创建患者ID到索引的映射（用于GroupKFold）
    # 我们需要创建一个包含所有患者ID的数组，每个患者ID出现一次
    patient_ids = np.array(all_patients)
    
    # 使用GroupKFold进行划分
    # 注意：GroupKFold需要groups数组，长度等于样本数
    # 但这里我们只需要对患者ID进行划分，所以创建一个虚拟的样本数组
    # 然后使用患者ID作为groups
    group_kfold = GroupKFold(n_splits=n_splits)
    
    # 创建虚拟索引数组（长度等于患者数）
    X_dummy = np.arange(len(patient_ids))
    groups = patient_ids  # 使用患者ID作为groups
    
    # 进行K-Fold划分
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(group_kfold.split(X_dummy, groups=groups)):
        train_patients = patient_ids[train_idx]
        val_patients = patient_ids[val_idx]
        
        print(f"\nFold {fold_idx + 1}:")
        print(f"  训练集患者数: {len(train_patients)}")
        print(f"  验证集患者数: {len(val_patients)}")
        
        # 根据患者ID划分40X数据
        train_df_40x = df_40x[df_40x['patient_id'].isin(train_patients)].copy()
        val_df_40x = df_40x[df_40x['patient_id'].isin(val_patients)].copy()
        
        # 根据患者ID划分100X数据
        train_df_100x = df_100x[df_100x['patient_id'].isin(train_patients)].copy()
        val_df_100x = df_100x[df_100x['patient_id'].isin(val_patients)].copy()
        
        print(f"  训练集: 40X={len(train_df_40x)}, 100X={len(train_df_100x)}")
        print(f"  验证集: 40X={len(val_df_40x)}, 100X={len(val_df_100x)}")
        
        # 保存CSV文件
        os.makedirs(output_dir, exist_ok=True)
        
        train_40x_path = os.path.join(output_dir, f"fold_{fold_idx + 1}_train_40x.csv")
        val_40x_path = os.path.join(output_dir, f"fold_{fold_idx + 1}_val_40x.csv")
        train_100x_path = os.path.join(output_dir, f"fold_{fold_idx + 1}_train_100x.csv")
        val_100x_path = os.path.join(output_dir, f"fold_{fold_idx + 1}_val_100x.csv")
        
        train_df_40x.to_csv(train_40x_path, index=False)
        val_df_40x.to_csv(val_40x_path, index=False)
        train_df_100x.to_csv(train_100x_path, index=False)
        val_df_100x.to_csv(val_100x_path, index=False)
        
        folds.append({
            'fold': fold_idx + 1,
            'train_40x': train_40x_path,
            'val_40x': val_40x_path,
            'train_100x': train_100x_path,
            'val_100x': val_100x_path,
            'train_patients': train_patients.tolist(),
            'val_patients': val_patients.tolist()
        })
    
    print(f"\n{'='*60}")
    print(f"K-Fold划分完成！所有CSV文件已保存到: {output_dir}")
    print(f"{'='*60}\n")
    
    return folds


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description="患者级K-Fold交叉验证划分")
    parser.add_argument('--csv_40x', type=str, required=True, help='40X图像CSV文件路径')
    parser.add_argument('--csv_100x', type=str, required=True, help='100X图像CSV文件路径')
    parser.add_argument('--n_splits', type=int, default=5, help='K-Fold折数')
    parser.add_argument('--output_dir', type=str, default='./kfold_splits', help='输出目录')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    folds = patient_level_kfold_split(
        csv_40x=args.csv_40x,
        csv_100x=args.csv_100x,
        n_splits=args.n_splits,
        random_state=args.random_state,
        output_dir=args.output_dir
    )
    
    print(f"\n生成了 {len(folds)} 个fold的划分")

