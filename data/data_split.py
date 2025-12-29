"""
数据集划分模块
根据病人ID进行分组划分，确保同一病人的图片不会跨数据集
"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os
import glob
import re
from pathlib import Path
import numpy as np


def split_dataset(dataset_root, output_dir="./", test_size=0.15, val_size=0.176, random_state=42):
    """
    根据病人ID划分数据集为训练集、验证集和测试集
    
    Args:
        dataset_root: 数据集根目录
        output_dir: 输出CSV文件的目录
        test_size: 测试集比例
        val_size: 验证集比例（相对于train+val的比例）
        random_state: 随机种子
    
    Returns:
        train_df, val_df, test_df: 三个DataFrame
    """
    print(f"正在从 {dataset_root} 读取图片...")
    
    # 1. 读取所有图片路径（支持多后缀；优先走 pathlib，兼容更多数据集结构）
    root = Path(dataset_root)
    if not root.exists():
        raise ValueError(f"数据集目录不存在: {dataset_root}")

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    all_image_paths = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    
    if len(all_image_paths) == 0:
        raise ValueError(f"在 {dataset_root} 中未找到任何图片文件（{sorted(exts)}）！")

    def _is_magnification_folder(name: str) -> bool:
        # 40X / 100X / 200X / 400X / 10X 等
        return bool(re.fullmatch(r"\d+X", name, flags=re.IGNORECASE))

    def _infer_class_from_path(img_path: str) -> str:
        """
        推断多分类 class（尽量兼容两种结构）：
        - dataset_cancer_v1: .../<MAG>X/<class>/<img>.png  -> parent 即 class
        - BreaKHis_v1: .../SOB/<class>/<patient>/<MAG>X/<img>.png
            这种情况下 parent 是倍率目录，需要跳过 patient 目录，取 SOB 下的 class
        """
        p = Path(img_path)
        parent = p.parent

        # 情况1：图片直接在 class 目录下
        if not _is_magnification_folder(parent.name):
            return parent.name

        # 情况2：parent 是倍率目录（40X/100X/...）
        # 可能是 .../<patient>/<MAG>X/<img> 或 .../<class>/<MAG>X/<img>
        up1 = parent.parent  # patient 或 class
        # 若上一级看起来像 BreaKHis 的 patient 文件夹（SOB_*），再往上取 class
        if up1.name.startswith("SOB_") and up1.parent is not None:
            return up1.parent.name
        return up1.name

    def _infer_binary_label(img_path: str, filename: str) -> str:
        low = img_path.lower()
        if "benign" in low:
            return "Benign"
        if "malignant" in low:
            return "Malignant"
        # 备用：BreaKHis / dataset_cancer_v1 文件名里常含 SOB_B_ / SOB_M_
        if "SOB_B_" in filename:
            return "Benign"
        if "SOB_M_" in filename:
            return "Malignant"
        return "Unknown"

    def _infer_patient_id(img_path: str, filename: str) -> str:
        """
        用于 GroupShuffleSplit 的 patient_id 推断：
        - BreaKHis / dataset_cancer_v1 常见：SOB_B_TA-14-16184CD-100-021.png -> 14-16184CD
        - 若不符合，退化为“文件名前缀（去后缀）”
        """
        stem = Path(filename).stem
        parts = stem.split('-')
        if len(parts) >= 3:
            return f"{parts[1]}-{parts[2]}"
        # 备用：路径里若出现类似 14-12345 的片段
        m = re.search(r"\b\d{2}-[0-9A-Za-z]+\b", img_path)
        if m:
            return m.group(0)
        return stem

    def _infer_magnification(img_path: str, filename: str) -> str:
        # 优先从父目录找到 40X/100X/...；否则从文件名的第4段
        p = Path(img_path)
        if _is_magnification_folder(p.parent.name):
            return p.parent.name
        stem = Path(filename).stem
        parts = stem.split('-')
        if len(parts) > 3:
            return parts[3]
        return "unknown"
    
    data = []
    for path in all_image_paths:
        filename = os.path.basename(path)

        patient_id = _infer_patient_id(path, filename)
        label = _infer_class_from_path(path)
        binary_label = _infer_binary_label(path, filename)
        mag = _infer_magnification(path, filename)
        
        data.append({
            "path": path,
            "patient_id": patient_id,
            "class": label,
            "binary_label": binary_label,
            "magnification": mag,
            "filename": filename
        })
    
    df = pd.DataFrame(data)
    
    print(f"总图片数: {len(df)}")
    print(f"总病人数: {df['patient_id'].nunique()}")
    print(f"类别分布:\n{df['class'].value_counts()}")
    
    # ---------------------------------------------------------
    # 2. 开始划分 (Train / Val / Test)
    # 目标：同一 patient_id 不跨集合 + 尽量保证每个集合都包含全部类别（8类）
    # 方法：先在“病人级别”做分层划分（patient -> class），再映射回图片级别
    # ---------------------------------------------------------

    rng = np.random.RandomState(random_state)

    # 病人级别标签（理论上每个病人只属于一个 class；若不一致，取众数并给出提示）
    patient_to_class = df.groupby("patient_id")["class"].agg(lambda x: x.mode().iloc[0])
    # 检查是否存在“同一病人多类别”的异常
    patient_class_nunique = df.groupby("patient_id")["class"].nunique()
    inconsistent = patient_class_nunique[patient_class_nunique > 1]
    if len(inconsistent) > 0:
        print(f"⚠️  警告：发现 {len(inconsistent)} 个 patient_id 对应多个 class，将使用众数作为病人标签")

    total_patients = len(patient_to_class)
    class_counts_pat = patient_to_class.value_counts()
    classes = sorted(class_counts_pat.index.tolist())

    # 期望的病人数量（按病人而不是按图片）
    test_target = int(round(total_patients * test_size))
    test_target = max(1, min(total_patients - 2, test_target))
    remaining = total_patients - test_target
    val_target = int(round(remaining * val_size))
    val_target = max(1, min(remaining - 1, val_target))

    # 将病人按类别分桶并打乱
    patients_by_class = {}
    for c in classes:
        pts = patient_to_class[patient_to_class == c].index.tolist()
        rng.shuffle(pts)
        patients_by_class[c] = pts

    train_patients, val_patients, test_patients = set(), set(), set()

    # 先确保“每个类别至少在 val/test 各出现 1 个病人”（如果该类病人数足够）
    # n>=3 才能保证 train/val/test 三个集合都至少有 1 个病人
    hard_limit_classes = []
    for c, pts in patients_by_class.items():
        if len(pts) >= 3:
            test_patients.add(pts.pop())
            val_patients.add(pts.pop())
            train_patients.update(pts)
        else:
            hard_limit_classes.append(c)
            # 病人数不足，无法保证三个集合都覆盖该类；优先保证训练集有该类
            train_patients.update(pts)

    if hard_limit_classes:
        print("⚠️  警告：以下类别病人数 < 3，无法保证 train/val/test 都包含该类：")
        for c in hard_limit_classes:
            print(f"  - {c}: {class_counts_pat[c]} patients")

    def _desired_counts(target_n: int):
        # 按类别占比计算目标数量，并对 n>=3 的类强制至少 1
        frac = class_counts_pat / float(total_patients) * float(target_n)
        base = frac.astype(float).apply(np.floor).astype(int)
        for c in classes:
            if class_counts_pat[c] >= 3:
                base[c] = max(base[c], 1)
        # 调整使总和等于 target_n
        diff = int(target_n - base.sum())
        # 依据小数部分排序
        order = (frac - np.floor(frac)).sort_values(ascending=False).index.tolist()
        if diff > 0:
            for c in order:
                if diff == 0:
                    break
                # 不能超过该类病人总数，也要留至少 1 给 train（若该类 >=3）
                max_allow = class_counts_pat[c] - (1 if class_counts_pat[c] >= 3 else 0)
                if base[c] < max_allow:
                    base[c] += 1
                    diff -= 1
        elif diff < 0:
            for c in reversed(order):
                if diff == 0:
                    break
                if base[c] > (1 if class_counts_pat[c] >= 3 else 0):
                    base[c] -= 1
                    diff += 1
        return base

    desired_test = _desired_counts(test_target)
    desired_val = _desired_counts(val_target)

    # 当前 train 可供移动的病人（按类）
    train_by_class = {c: [pid for pid in train_patients if patient_to_class[pid] == c] for c in classes}
    for c in classes:
        rng.shuffle(train_by_class[c])

    def _move_from_train(dst_set: set, desired: pd.Series, name: str):
        # 将 train 中的病人移动到 dst_set，尽量满足各类别 desired
        current = {c: 0 for c in classes}
        for pid in dst_set:
            current[patient_to_class[pid]] += 1

        # 先按类别补齐
        for c in classes:
            while current[c] < int(desired[c]):
                if not train_by_class[c]:
                    break
                # 保证该类 train 里至少留 1 个病人（若该类 >=3）
                if class_counts_pat[c] >= 3:
                    remaining_in_train = sum(1 for pid in train_patients if patient_to_class[pid] == c)
                    if remaining_in_train <= 1:
                        break
                pid = train_by_class[c].pop()
                if pid in train_patients:
                    train_patients.remove(pid)
                    dst_set.add(pid)
                    current[c] += 1

        # 若总数仍不足，做兜底补齐
        while len(dst_set) < int(desired.sum()):
            # 找一个当前 train 里“最多”的类别来补
            candidates = []
            for c in classes:
                if not train_by_class[c]:
                    continue
                if class_counts_pat[c] >= 3:
                    remaining_in_train = sum(1 for pid in train_patients if patient_to_class[pid] == c)
                    if remaining_in_train <= 1:
                        continue
                candidates.append(c)
            if not candidates:
                break
            # 选择 train 里剩余最多的类
            candidates.sort(key=lambda c: sum(1 for pid in train_patients if patient_to_class[pid] == c), reverse=True)
            c = candidates[0]
            pid = train_by_class[c].pop()
            if pid in train_patients:
                train_patients.remove(pid)
                dst_set.add(pid)

        if len(dst_set) != int(desired.sum()):
            print(f"⚠️  警告：{name} 病人数期望 {int(desired.sum())}，实际 {len(dst_set)}（受限于类别病人数/约束）")

    # 先补齐 test，再补齐 val（避免互相抢同一批病人）
    _move_from_train(test_patients, desired_test, "Test")
    _move_from_train(val_patients, desired_val, "Validation")

    # 剩余全部归 train
    #（train_patients 已在移动过程中维护）

    train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_patients)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_patients)].reset_index(drop=True)
    
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
    
    # 4. 保存为CSV
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_split.csv")
    val_path = os.path.join(output_dir, "val_split.csv")
    test_path = os.path.join(output_dir, "test_split.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✓ 数据集划分完成！CSV文件已保存到 {output_dir}")
    
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
    # 使用示例
    dataset_root = "path/to/your/dataset"  # 修改为实际路径
    split_dataset(dataset_root, output_dir="./")

