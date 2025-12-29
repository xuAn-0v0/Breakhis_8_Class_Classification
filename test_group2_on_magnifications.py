#!/usr/bin/env python
"""
使用 group2 (result_20251217_192008) 模型在不同放大倍数测试集上评估
测试三次：40x, 100x, 混合(40x+100x)
"""
import os
import sys
import yaml
import torch
import json
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from data.dataset import CancerDataset, get_transforms
from models import get_ctranspath_legacy, LEGACY_AVAILABLE
from utils.train_utils import validate
from utils.metrics import calculate_metrics, plot_confusion_matrix, save_predictions

def load_config_for_test(test_csv, experiment_suffix):
    """创建测试配置"""
    config = {
        'experiment_name': f'group2_ctrans_linear_frozen_{experiment_suffix}',
        'group': 'group2',
        'model': {
            'name': 'ctranspath',
            'num_classes': 8,
            'pretrained_path': 'ctranspath.pth',
            'classifier_type': 'linear',
            'freeze_backbone': True
        },
        'data': {
            'dataset_root': 'BreaKHis_v1/BreaKHis_v1/histology_slides/breast',
            'test_csv': test_csv,
            'image_size': 224,
            'batch_size': 32,
            'num_workers': 4
        },
        'results': {
            'save_dir': f'result_magnification_test/results/group2_{experiment_suffix}',
            'save_predictions': True,
            'plot_confusion_matrix': True
        },
        'device': 'cuda'
    }
    return config

def run_test(config, checkpoint_path, device):
    """运行单次测试"""
    print(f"\n{'='*60}")
    print(f"Testing: {config['experiment_name']}")
    print(f"{'='*60}\n")
    
    # 创建测试数据集
    print("Loading test dataset...")
    test_transform = get_transforms(
        image_size=config['data']['image_size'],
        is_training=False
    )
    
    test_dataset = CancerDataset(
        csv_file=config['data']['test_csv'],
        transform=test_transform,
        num_classes=config['model']['num_classes']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    print(f"✓ Test set: {len(test_dataset)} images\n")
    
    # 构建模型
    print("Building model...")
    if not LEGACY_AVAILABLE:
        print("ERROR: Legacy implementation not available!")
        return None
    
    model = get_ctranspath_legacy(
        num_classes=config['model']['num_classes'],
        pretrained_path=config['model'].get('pretrained_path', 'ctranspath.pth'),
        classifier_type=config['model'].get('classifier_type', 'linear'),
        freeze_backbone=config['model'].get('freeze_backbone', True)
    )
    model = model.to(device)
    
    # 加载训练好的权重
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}\n")
    
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 开始测试
    print(f"{'='*60}")
    print("Start Testing")
    print(f"{'='*60}\n")
    
    test_loss, test_acc, all_preds, all_labels, all_probs = validate(
        model, test_loader, criterion, device, phase='Test'
    )
    
    print(f"\n{'='*60}")
    print(f"Test Results")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # 计算详细指标
    class_names = test_dataset.class_names
    
    # 获取二分类标签
    benign_classes = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma']
    malignant_classes = ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
    
    binary_labels_true = []
    binary_labels_pred = []
    
    for label in all_labels:
        class_name = class_names[label]
        if class_name in benign_classes:
            binary_labels_true.append('Benign')
        elif class_name in malignant_classes:
            binary_labels_true.append('Malignant')
        else:
            binary_labels_true.append('Benign')
    
    for pred in all_preds:
        class_name = class_names[pred]
        if class_name in benign_classes:
            binary_labels_pred.append('Benign')
        elif class_name in malignant_classes:
            binary_labels_pred.append('Malignant')
        else:
            binary_labels_pred.append('Benign')
    
    # 计算指标
    metrics = calculate_metrics(all_labels, all_preds, class_names, 
                                binary_labels_true, binary_labels_pred)
    
    # 保存混淆矩阵
    if config['results'].get('plot_confusion_matrix', True):
        results_dir = config['results']['save_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        cm_path = os.path.join(results_dir, 'confusion_matrix.png')
        plot_confusion_matrix(all_labels, all_preds, class_names, save_path=cm_path)
        
        cm_norm_path = os.path.join(results_dir, 'confusion_matrix_normalized.png')
        plot_confusion_matrix(
            all_labels, all_preds, class_names, 
            save_path=cm_norm_path, normalize=True
        )
    
    # 保存预测结果
    if config['results'].get('save_predictions', True):
        image_paths = [test_dataset.df.iloc[i]['path'] for i in range(len(test_dataset))]
        pred_path = os.path.join(results_dir, 'predictions.csv')
        save_predictions(
            all_labels, all_preds, all_probs, 
            image_paths, class_names, pred_path
        )
    
    # 保存指标到JSON
    metrics_path = os.path.join(results_dir, 'test_metrics.json')
    test_results = {
        'experiment_name': config['experiment_name'],
        'group': config['group'],
        'model_name': config['model']['name'],
        'checkpoint': checkpoint_path,
        'test_csv': config['data']['test_csv'],
        'num_samples': len(test_dataset),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'f1_macro': float(metrics['f1_macro']),
            'f1_weighted': float(metrics['f1_weighted']),
            'precision_macro': float(metrics['precision_macro']),
            'precision_weighted': float(metrics['precision_weighted']),
            'recall_macro': float(metrics['recall_macro']),
            'recall_weighted': float(metrics['recall_weighted'])
        }
    }
    
    if 'binary_accuracy' in metrics:
        test_results['binary_metrics'] = {
            'binary_accuracy': float(metrics['binary_accuracy']),
            'binary_f1': float(metrics['binary_f1']),
            'binary_precision': float(metrics['binary_precision']),
            'binary_recall': float(metrics['binary_recall'])
        }
    
    with open(metrics_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    
    print(f"\n✓ Results saved to {results_dir}")
    print(f"{'='*60}\n")
    
    return test_results


def main():
    print("="*80)
    print("Group2 模型多放大倍数测试评估")
    print("="*80)
    print(f"\nCheckpoint: result_20251217_192008/checkpoints/group2_ctrans_linear_frozen_breakhis/best_model.pth")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Checkpoint路径
    checkpoint_path = 'result_20251217_192008/checkpoints/group2_ctrans_linear_frozen_breakhis/best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    # 测试配置
    test_configs = [
        ('ai4bio-project-Breakhis/test_split_40x.csv', '40x', '40x 放大倍数 (305 samples)'),
        ('ai4bio-project-Breakhis/test_split_100x.csv', '100x', '100x 放大倍数 (305 samples)'),
        ('ai4bio-project-Breakhis/test_split_40x_100x_mixed.csv', 'mixed', '混合 40x+100x (610 samples)')
    ]
    
    all_results = {}
    
    for test_csv, suffix, description in test_configs:
        print("\n" + "="*80)
        print(f"测试 {description}")
        print("="*80)
        
        if not os.path.exists(test_csv):
            print(f"WARNING: Test CSV not found: {test_csv}")
            continue
        
        # 创建配置
        config = load_config_for_test(test_csv, suffix)
        
        # 运行测试
        result = run_test(config, checkpoint_path, device)
        if result:
            all_results[suffix] = result
    
    # 打印所有结果总结
    print("\n" + "="*80)
    print("所有测试完成 - 结果总结")
    print("="*80)
    
    for suffix, result in all_results.items():
        print(f"\n{suffix.upper()}:")
        print(f"  样本数: {result['num_samples']}")
        print(f"  8分类准确率: {result['test_accuracy']:.2f}%")
        print(f"  F1-Score (Macro): {result['metrics']['f1_macro']:.4f}")
        if 'binary_metrics' in result:
            print(f"  二分类准确率: {result['binary_metrics']['binary_accuracy']*100:.2f}%")
            print(f"  二分类F1: {result['binary_metrics']['binary_f1']:.4f}")
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80)
    print("✅ 所有评估完成！")
    print("="*80)
    print("\n结果保存在: result_magnification_test/results/")
    print("  - group2_40x/")
    print("  - group2_100x/")
    print("  - group2_mixed/")


if __name__ == '__main__':
    main()
