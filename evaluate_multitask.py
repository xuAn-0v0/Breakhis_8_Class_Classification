"""
通用多任务学习评估脚本
在40x和100x测试集上分别评估2分类和8分类性能，并计算混合(Mixed)结果
"""
import os
import sys
import argparse
import yaml
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__)) if os.path.dirname(os.path.abspath(__file__)) else '/songjian/zixuan'
sys.path.insert(0, script_dir)  # 添加主目录
sys.path.insert(0, os.path.join(script_dir, 'ai4bio-project-Breakhis'))

from data.multitask_dataset import MultiTaskDataset
from utils.metrics import plot_confusion_matrix as plot_cm

# 导入模型
from models.simple_cnn_multitask import SimpleCNNMultiTask
from models.ctranspath_multitask import get_ctranspath_multitask_model
from models.timm_multitask import get_timm_multitask_model


def get_transforms(image_size=224):
    """获取测试数据变换"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_model(config):
    """根据配置创建模型"""
    model_cfg = config['model']
    model_name = model_cfg['name']
    
    if model_name == 'simple_cnn_multitask':
        model = SimpleCNNMultiTask(
            num_classes_binary=model_cfg['num_classes_binary'],
            num_classes_multiclass=model_cfg['num_classes_multiclass'],
            dropout=model_cfg.get('dropout', 0.5),
            use_multiclass=model_cfg.get('use_multiclass', True)
        )
    elif model_name == 'ctranspath_multitask':
        model = get_ctranspath_multitask_model(
            num_classes_binary=model_cfg['num_classes_binary'],
            num_classes_multiclass=model_cfg['num_classes_multiclass'],
            group=config['group'],
            ctranspath_weight_path=model_cfg.get('pretrained_path', 'ctranspath.pth'),
            use_multiclass=model_cfg.get('use_multiclass', True)
        )
    elif model_name == 'timm_multitask':
        model = get_timm_multitask_model(
            model_name=model_cfg.get('backbone', 'resnet50'),
            num_classes_binary=model_cfg['num_classes_binary'],
            num_classes_multiclass=model_cfg['num_classes_multiclass'],
            dropout=model_cfg.get('dropout', 0.5),
            use_multiclass=model_cfg.get('use_multiclass', True),
            pretrained=model_cfg.get('pretrained', True)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def evaluate_model(model, dataloader, device, use_multiclass):
    """评估模型"""
    model.eval()
    
    all_labels_binary = []
    all_preds_binary = []
    
    all_labels_multiclass = []
    all_preds_multiclass = []
    
    with torch.no_grad():
        for images, labels_binary, labels_multiclass in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 二分类
            probs_binary = torch.softmax(outputs['logits_binary'], dim=1)
            _, preds_binary = torch.max(probs_binary, 1)
            
            all_labels_binary.extend(labels_binary.cpu().numpy())
            all_preds_binary.extend(preds_binary.cpu().numpy())
            
            # 八分类
            if use_multiclass and outputs['logits_multiclass'] is not None:
                probs_multiclass = torch.softmax(outputs['logits_multiclass'], dim=1)
                _, preds_multiclass = torch.max(probs_multiclass, 1)
                
                all_labels_multiclass.extend(labels_multiclass.cpu().numpy())
                all_preds_multiclass.extend(preds_multiclass.cpu().numpy())
    
    return {
        'binary': {
            'labels': np.array(all_labels_binary),
            'preds': np.array(all_preds_binary)
        },
        'multiclass': {
            'labels': np.array(all_labels_multiclass),
            'preds': np.array(all_preds_multiclass)
        } if use_multiclass else None
    }


def compute_metrics(labels, preds, class_names):
    """计算评估指标"""
    if len(labels) == 0:
        return None
        
    accuracy = accuracy_score(labels, preds) * 100
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro * 100),
        'f1_weighted': float(f1_weighted * 100),
        'precision_weighted': float(precision_weighted * 100),
        'recall_weighted': float(recall_weighted * 100)
    }
    return metrics


def print_metrics(metrics, class_names, task_name):
    """打印评估指标"""
    if not metrics: return
    print(f"\n{task_name} 结果: Acc: {metrics['accuracy']:.2f}%, F1(W): {metrics['f1_weighted']:.2f}%")


def main(args):
    # 确定配置文件路径
    config_path = args.config
    checkpoint_dir = args.checkpoint_dir.rstrip('/')
    
    if not config_path:
        # 尝试更多可能的路径
        exp_name = os.path.basename(checkpoint_dir)
        potential_paths = [
            os.path.join(checkpoint_dir, 'config.yaml'),
            os.path.join(os.path.dirname(checkpoint_dir), '..', 'results', exp_name, 'config.yaml'),
            os.path.join('configs', f"{exp_name}.yaml"),
            os.path.join('configs', f"multitask_{exp_name}.yaml"),
            os.path.join('configs', exp_name.replace('multitask_', '') + '.yaml'),
        ]
        # 如果 exp_name 包含 multitask_，也尝试去掉它再找
        if exp_name.startswith('multitask_'):
            potential_paths.append(os.path.join(os.path.dirname(checkpoint_dir), '..', 'results', exp_name.replace('multitask_', ''), 'config.yaml'))
        
        for p in potential_paths:
            if os.path.exists(p):
                config_path = p
                break
    
    if not config_path or not os.path.exists(config_path):
        print(f"Error: 找不到配置文件。")
        print(f"尝试过的路径包括: {potential_paths if not args.config else config_path}")
        print(f"请使用 --config 手动指定配置文件路径。")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config).to(device)
    
    # 加载权重
    pth_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if not os.path.exists(pth_path):
        # 尝试直接作为路径
        pth_path = args.checkpoint_dir if args.checkpoint_dir.endswith('.pth') else pth_path
        
    checkpoint = torch.load(pth_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    data_cfg = config['data']
    transform = get_transforms(data_cfg.get('image_size', 224))
    use_multiclass = config['model'].get('use_multiclass', True)
    
    class_names_binary = ['Benign', 'Malignant']
    class_names_multiclass = ['A', 'DC', 'F', 'LC', 'MC', 'PC', 'PT', 'TA']
    
    test_mags = [('40x', data_cfg['test_csv_40x']), ('100x', data_cfg['test_csv_100x'])]
    all_results = {}
    comb_bin_labels, comb_bin_preds = [], []
    comb_multi_labels, comb_multi_preds = [], []

    # 获取结果保存基目录
    results_base_dir = config.get('results', {}).get('save_dir', os.path.join('multitask_results', 'results', config['experiment_name']))
    
    for mag_name, csv_path in test_mags:
        ds = MultiTaskDataset(csv_file=csv_path, root_dir=data_cfg['dataset_root'], transform=transform)
        loader = DataLoader(ds, batch_size=data_cfg.get('batch_size', 32), shuffle=False, num_workers=4)
        
        res = evaluate_model(model, loader, device, use_multiclass)
        
        bin_metrics = compute_metrics(res['binary']['labels'], res['binary']['preds'], class_names_binary)
        print_metrics(bin_metrics, class_names_binary, f"{mag_name} Binary")
        
        multi_metrics = None
        if use_multiclass:
            multi_metrics = compute_metrics(res['multiclass']['labels'], res['multiclass']['preds'], class_names_multiclass)
            print_metrics(multi_metrics, class_names_multiclass, f"{mag_name} 8-class")
            
        # 保存单个结果
        res_dir = os.path.join(results_base_dir, mag_name)
        os.makedirs(res_dir, exist_ok=True)
        
        # 绘制并保存混淆矩阵
        plot_cm(res['binary']['labels'], res['binary']['preds'], class_names_binary, 
                save_path=os.path.join(res_dir, 'confusion_matrix_binary.png'))
        if use_multiclass:
            plot_cm(res['multiclass']['labels'], res['multiclass']['preds'], class_names_multiclass, 
                    save_path=os.path.join(res_dir, 'confusion_matrix_multiclass.png'))
            # 归一化版本
            plot_cm(res['multiclass']['labels'], res['multiclass']['preds'], class_names_multiclass, 
                    save_path=os.path.join(res_dir, 'confusion_matrix_multiclass_norm.png'), normalize=True)

        with open(os.path.join(res_dir, 'test_metrics.json'), 'w') as f:
            json.dump({'binary': bin_metrics, 'multiclass': multi_metrics}, f, indent=2)
            
        comb_bin_labels.extend(res['binary']['labels'])
        comb_bin_preds.extend(res['binary']['preds'])
        if use_multiclass:
            comb_multi_labels.extend(res['multiclass']['labels'])
            comb_multi_preds.extend(res['multiclass']['preds'])
            
        all_results[mag_name] = {'binary': bin_metrics, 'multiclass': multi_metrics}

    # Mixed (Combined)
    mixed_bin = compute_metrics(np.array(comb_bin_labels), np.array(comb_bin_preds), class_names_binary)
    mixed_multi = compute_metrics(np.array(comb_multi_labels), np.array(comb_multi_preds), class_names_multiclass)
    print_metrics(mixed_bin, class_names_binary, "Mixed Binary")
    print_metrics(mixed_multi, class_names_multiclass, "Mixed 8-class")
    
    res_dir = os.path.join(results_base_dir, 'mixed')
    os.makedirs(res_dir, exist_ok=True)
    
    # 绘制并保存混合结果的混淆矩阵
    plot_cm(np.array(comb_bin_labels), np.array(comb_bin_preds), class_names_binary, 
            save_path=os.path.join(res_dir, 'confusion_matrix_binary.png'))
    if use_multiclass:
        plot_cm(np.array(comb_multi_labels), np.array(comb_multi_preds), class_names_multiclass, 
                save_path=os.path.join(res_dir, 'confusion_matrix_multiclass.png'))
        plot_cm(np.array(comb_multi_labels), np.array(comb_multi_preds), class_names_multiclass, 
                save_path=os.path.join(res_dir, 'confusion_matrix_multiclass_norm.png'), normalize=True)

    with open(os.path.join(res_dir, 'test_metrics.json'), 'w') as f:
        json.dump({'binary': mixed_bin, 'multiclass': mixed_multi}, f, indent=2)
    
    all_results['mixed'] = {'binary': mixed_bin, 'multiclass': mixed_multi}
    with open(os.path.join(results_base_dir, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default=None)
    main(parser.parse_args())
