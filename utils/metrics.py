"""
评估指标相关工具函数
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import os


def calculate_metrics(y_true, y_pred, class_names=None, binary_labels_true=None, binary_labels_pred=None):
    """
    计算各种评估指标
    
    Args:
        y_true: 真实标签 (8分类)
        y_pred: 预测标签 (8分类)
        class_names: 类别名称列表
        binary_labels_true: 真实的二分类标签 (可选)
        binary_labels_pred: 预测的二分类标签 (可选)
    
    Returns:
        包含各种指标的字典
    """
    # 8分类指标
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # 计算二分类指标（如果提供了二分类标签）
    if binary_labels_true is not None and binary_labels_pred is not None:
        metrics['binary_accuracy'] = accuracy_score(binary_labels_true, binary_labels_pred)
        metrics['binary_precision'] = precision_score(binary_labels_true, binary_labels_pred, 
                                                      pos_label='Malignant', zero_division=0)
        metrics['binary_recall'] = recall_score(binary_labels_true, binary_labels_pred, 
                                                pos_label='Malignant', zero_division=0)
        metrics['binary_f1'] = f1_score(binary_labels_true, binary_labels_pred, 
                                       pos_label='Malignant', zero_division=0)
    
    # 打印8分类报告
    print("\n" + "="*60)
    print("8-Class Classification Report:")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # 打印8分类总体指标
    print("\n" + "="*60)
    print("8-Class Overall Metrics:")
    print("="*60)
    print(f"{'Accuracy':20s}: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"{'F1-Score (Macro)':20s}: {metrics['f1_macro']:.4f}")
    print(f"{'F1-Score (Weighted)':20s}: {metrics['f1_weighted']:.4f}")
    print(f"{'Precision (Macro)':20s}: {metrics['precision_macro']:.4f}")
    print(f"{'Precision (Weighted)':20s}: {metrics['precision_weighted']:.4f}")
    print(f"{'Recall (Macro)':20s}: {metrics['recall_macro']:.4f}")
    print(f"{'Recall (Weighted)':20s}: {metrics['recall_weighted']:.4f}")
    
    # 打印二分类指标
    if binary_labels_true is not None and binary_labels_pred is not None:
        print("\n" + "="*60)
        print("Binary Classification (Benign vs Malignant) Metrics:")
        print("="*60)
        print(f"{'Binary Accuracy':20s}: {metrics['binary_accuracy']:.4f} ({metrics['binary_accuracy']*100:.2f}%)")
        print(f"{'Binary F1-Score':20s}: {metrics['binary_f1']:.4f}")
        print(f"{'Binary Precision':20s}: {metrics['binary_precision']:.4f}")
        print(f"{'Binary Recall':20s}: {metrics['binary_recall']:.4f}")
        
        # 打印二分类详细报告
        print("\n" + "="*60)
        print("Binary Classification Report:")
        print("="*60)
        print(classification_report(binary_labels_true, binary_labels_pred, 
                                   target_names=['Benign', 'Malignant'], zero_division=0))
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalize=False):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
        normalize: 是否归一化
    """
    # 确保混淆矩阵包含所有类别，即使某些类别在预测中没有出现
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史字典，包含 'train_loss', 'train_acc', 'val_loss', 'val_acc'
                 或者多任务版本的 'train_acc_binary', 'val_acc_binary' 等
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy曲线 - 兼容多任务版本
    if 'train_acc' in history:
        train_acc = history['train_acc']
        val_acc = history['val_acc']
        label_suffix = ""
    elif 'train_acc_binary' in history:
        train_acc = history['train_acc_binary']
        val_acc = history['val_acc_binary']
        label_suffix = " (Binary)"
    else:
        # 如果都没有，尝试取第一个包含 acc 的键
        acc_keys = [k for k in history.keys() if 'acc' in k and 'train' in k]
        if acc_keys:
            train_acc = history[acc_keys[0]]
            val_key = acc_keys[0].replace('train', 'val')
            val_acc = history[val_key] if val_key in history else train_acc
            label_suffix = f" ({acc_keys[0].replace('train_acc_', '')})"
        else:
            print("Warning: No accuracy metrics found in history for plotting.")
            train_acc = [0] * len(epochs)
            val_acc = [0] * len(epochs)
            label_suffix = ""

    ax2.plot(epochs, train_acc, 'b-', label=f'Train Acc{label_suffix}', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-', label=f'Val Acc{label_suffix}', linewidth=2)
    
    # 如果有多任务的八分类准确率，也画出来
    if 'train_acc_multiclass' in history:
        ax2.plot(epochs, history['train_acc_multiclass'], 'g--', label='Train Acc (8-class)', linewidth=1)
        ax2.plot(epochs, history['val_acc_multiclass'], 'm--', label='Val Acc (8-class)', linewidth=1)

    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved to {save_path}")
    
    plt.close()


def save_predictions(y_true, y_pred, y_probs, image_paths, class_names, save_path):
    """
    保存预测结果到CSV文件
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_probs: 预测概率
        image_paths: 图像路径列表
        class_names: 类别名称列表
        save_path: 保存路径
    """
    import pandas as pd
    
    results = []
    for i in range(len(y_true)):
        result = {
            'image_path': image_paths[i],
            'true_label': class_names[y_true[i]],
            'pred_label': class_names[y_pred[i]],
            'correct': y_true[i] == y_pred[i]
        }
        # 添加每个类别的概率
        for j, cls_name in enumerate(class_names):
            result[f'prob_{cls_name}'] = y_probs[i][j]
        
        results.append(result)
    
    df = pd.DataFrame(results)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✓ Predictions saved to {save_path}")

