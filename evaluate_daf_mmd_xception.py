"""
DAF-MMD Net with Xception 评估脚本
在测试集上评估训练好的模型，生成混淆矩阵和指标
"""
import argparse
import yaml
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support

from data.paired_dataset import PairedMagnificationDataset
from data.dataset import get_transforms
from models.daf_mmd_net_xception import get_daf_mmd_xception_model
from utils.metrics import plot_confusion_matrix


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(model, test_loader, device, class_names_binary, class_names_multiclass, use_multiclass=True):
    """评估模型"""
    model.eval()
    
    all_labels_binary = []
    all_preds_binary = []
    all_labels_multiclass = []
    all_preds_multiclass = []
    
    with torch.no_grad():
        for images_40x, images_100x, labels_binary, labels_multiclass, _ in test_loader:
            images_40x = images_40x.to(device)
            images_100x = images_100x.to(device)
            
            outputs = model(images_40x, images_100x)
            
            # 二分类
            probs_binary = torch.softmax(outputs['logits_final_binary'], dim=1)
            preds_binary = torch.argmax(probs_binary, dim=1)
            
            all_labels_binary.extend(labels_binary.cpu().numpy())
            all_preds_binary.extend(preds_binary.cpu().numpy())
            
            # 八分类
            if use_multiclass and outputs['logits_final_multiclass'] is not None:
                probs_multiclass = torch.softmax(outputs['logits_final_multiclass'], dim=1)
                preds_multiclass = torch.argmax(probs_multiclass, dim=1)
                
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
    
    return {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro * 100),
        'f1_weighted': float(f1_weighted * 100),
        'precision_weighted': float(precision_weighted * 100),
        'recall_weighted': float(recall_weighted * 100)
    }


def main(args):
    # 加载配置
    if args.config:
        config_path = args.config
    else:
        # 尝试从checkpoint目录找到配置
        config_path = os.path.join(args.checkpoint_dir, '..', 'results', os.path.basename(args.checkpoint_dir), 'config.yaml')
        if not os.path.exists(config_path):
            config_path = os.path.join(args.checkpoint_dir, '..', '..', 'results', os.path.basename(args.checkpoint_dir), 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    
    config = load_config(config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = get_daf_mmd_xception_model(
        stream_config=config.get('stream_config', 'xception_swin'),
        num_classes_binary=config['model']['num_classes_binary'],
        num_classes_multiclass=config['model']['num_classes_multiclass'],
        use_multiclass=config['model'].get('use_multiclass', True),
        use_pretrained=False  # 评估时不需要预训练权重
    )
    model = model.to(device)
    
    # 加载权重
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        # 尝试查找 last_model.pth
        last_model_path = os.path.join(args.checkpoint_dir, 'last_model.pth')
        if os.path.exists(last_model_path):
            print(f"⚠️  警告: 未找到 best_model.pth，尝试加载 last_model.pth")
            checkpoint_path = last_model_path
        else:
             raise FileNotFoundError(f"找不到checkpoint: {checkpoint_path} 或 {last_model_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 模型加载成功 (Epoch {checkpoint.get('epoch', 'N/A')})")
    
    # 创建测试数据集
    data_cfg = config['data']
    image_size_40x = data_cfg.get('image_size_40x', 299)
    image_size_100x = data_cfg.get('image_size_100x', 224)
    
    class_names_binary = ['Benign', 'Malignant']
    class_names_multiclass = ['A', 'DC', 'F', 'LC', 'MC', 'PC', 'PT', 'TA']
    use_multiclass = config['model'].get('use_multiclass', True)
    
    # 评估40x和100x测试集
    test_configs = [
        ('40x', data_cfg['test_csv_40x']),
        ('100x', data_cfg['test_csv_100x'])
    ]
    
    results_base_dir = config.get('results', {}).get('save_dir', os.path.join('multitask_results', 'results', config['experiment_name']))
    
    all_results = {}
    comb_bin_labels, comb_bin_preds = [], []
    comb_multi_labels, comb_multi_preds = [], []
    
    for test_name, csv_path in test_configs:
        print(f"\n评估 {test_name} 测试集...")
        
        test_dataset = PairedMagnificationDataset(
            csv_40x=csv_path,
            csv_100x=csv_path,  # 使用同一个CSV，但会分别加载40x和100x图像
            transform_40x=get_transforms(image_size=image_size_40x, is_training=False),
            transform_100x=get_transforms(image_size=image_size_100x, is_training=False)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=data_cfg.get('batch_size', 16),
            shuffle=False,
            num_workers=4
        )
        
        res = evaluate_model(model, test_loader, device, class_names_binary, class_names_multiclass, use_multiclass)
        
        bin_metrics = compute_metrics(res['binary']['labels'], res['binary']['preds'], class_names_binary)
        multi_metrics = compute_metrics(res['multiclass']['labels'], res['multiclass']['preds'], class_names_multiclass) if use_multiclass else None
        
        print(f"\n{test_name} 二分类结果:")
        print(f"  Accuracy: {bin_metrics['accuracy']:.2f}%")
        print(f"  F1-Macro: {bin_metrics['f1_macro']:.2f}%")
        
        if use_multiclass:
            print(f"\n{test_name} 八分类结果:")
            print(f"  Accuracy: {multi_metrics['accuracy']:.2f}%")
            print(f"  F1-Macro: {multi_metrics['f1_macro']:.2f}%")
        
        # 保存结果
        res_dir = os.path.join(results_base_dir, test_name)
        os.makedirs(res_dir, exist_ok=True)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(
            res['binary']['labels'], res['binary']['preds'], class_names_binary,
            save_path=os.path.join(res_dir, 'confusion_matrix_binary.png')
        )
        if use_multiclass:
            plot_confusion_matrix(
                res['multiclass']['labels'], res['multiclass']['preds'], class_names_multiclass,
                save_path=os.path.join(res_dir, 'confusion_matrix_multiclass.png')
            )
            plot_confusion_matrix(
                res['multiclass']['labels'], res['multiclass']['preds'], class_names_multiclass,
                save_path=os.path.join(res_dir, 'confusion_matrix_multiclass_norm.png'),
                normalize=True
            )
        
        with open(os.path.join(res_dir, 'test_metrics.json'), 'w') as f:
            json.dump({'binary': bin_metrics, 'multiclass': multi_metrics}, f, indent=2)
        
        all_results[test_name] = {'binary': bin_metrics, 'multiclass': multi_metrics}
        comb_bin_labels.extend(res['binary']['labels'])
        comb_bin_preds.extend(res['binary']['preds'])
        if use_multiclass:
            comb_multi_labels.extend(res['multiclass']['labels'])
            comb_multi_preds.extend(res['multiclass']['preds'])
    
    # Mixed结果
    mixed_bin = compute_metrics(np.array(comb_bin_labels), np.array(comb_bin_preds), class_names_binary)
    mixed_multi = compute_metrics(np.array(comb_multi_labels), np.array(comb_multi_preds), class_names_multiclass) if use_multiclass else None
    
    res_dir = os.path.join(results_base_dir, 'mixed')
    os.makedirs(res_dir, exist_ok=True)
    
    plot_confusion_matrix(
        np.array(comb_bin_labels), np.array(comb_bin_preds), class_names_binary,
        save_path=os.path.join(res_dir, 'confusion_matrix_binary.png')
    )
    if use_multiclass:
        plot_confusion_matrix(
            np.array(comb_multi_labels), np.array(comb_multi_preds), class_names_multiclass,
            save_path=os.path.join(res_dir, 'confusion_matrix_multiclass.png')
        )
        plot_confusion_matrix(
            np.array(comb_multi_labels), np.array(comb_multi_preds), class_names_multiclass,
            save_path=os.path.join(res_dir, 'confusion_matrix_multiclass_norm.png'),
            normalize=True
        )
    
    with open(os.path.join(res_dir, 'test_metrics.json'), 'w') as f:
        json.dump({'binary': mixed_bin, 'multiclass': mixed_multi}, f, indent=2)
    
    all_results['mixed'] = {'binary': mixed_bin, 'multiclass': mixed_multi}
    
    with open(os.path.join(results_base_dir, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ 评估完成，结果保存在 {results_base_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Checkpoint目录路径')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径（可选）')
    args = parser.parse_args()
    
    main(args)


