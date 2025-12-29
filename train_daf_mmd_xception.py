"""
DAF-MMD Net with Xception 训练脚本
支持Xception-Swin和Xception-Xception两种配置
所有早停和模型保存标准以八分类为准
"""
import argparse
import yaml
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json
from datetime import datetime

from data.paired_dataset import PairedMagnificationDataset
from data.dataset import get_transforms
from models.daf_mmd_net_xception import get_daf_mmd_xception_model
from utils.focal_loss import FocalLoss
from utils.mmd_loss import MMDLoss
from utils.metrics import plot_training_history


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_mmd_weight(epoch, total_epochs, initial_weight=0.05, final_weight=1.0, start_epoch_ratio=0.5):
    """计算当前epoch的MMD权重（递增策略）"""
    start_epoch = int(total_epochs * start_epoch_ratio)
    if epoch < start_epoch:
        return initial_weight
    else:
        progress = (epoch - start_epoch) / (total_epochs - start_epoch)
        weight = initial_weight + (final_weight - initial_weight) * progress
        return weight


def train_one_epoch(
    model, train_loader, criterion_binary, criterion_multiclass, criterion_mmd,
    optimizer, device, epoch, total_epochs, alpha_multiclass=1.0,
    mmd_initial_weight=0.05, mmd_final_weight=1.0, mmd_start_ratio=0.5,
    use_multiclass=True
):
    """训练一个epoch"""
    model.train()
    
    running_loss = 0.0
    running_loss_binary = 0.0
    running_loss_multiclass = 0.0
    running_loss_mmd = 0.0
    
    correct_binary = 0
    correct_multiclass = 0
    total_samples = 0
    
    mmd_weight = get_mmd_weight(epoch, total_epochs, mmd_initial_weight, mmd_final_weight, mmd_start_ratio)
    
    from tqdm import tqdm
    for images_40x, images_100x, labels_binary, labels_multiclass, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
        images_40x = images_40x.to(device)
        images_100x = images_100x.to(device)
        labels_binary = labels_binary.to(device)
        labels_multiclass = labels_multiclass.to(device)
        
        optimizer.zero_grad()
        outputs = model(images_40x, images_100x)
        
        # 二分类损失
        loss_binary = criterion_binary(outputs['logits_final_binary'], labels_binary)
        
        # 八分类损失
        loss_multiclass = torch.tensor(0.0).to(device)
        if use_multiclass and outputs['logits_final_multiclass'] is not None:
            loss_multiclass = criterion_multiclass(outputs['logits_final_multiclass'], labels_multiclass)
        
        # MMD损失
        loss_mmd = criterion_mmd(outputs['features_40x'], outputs['features_100x'])
        
        # 总损失
        total_loss = loss_binary + alpha_multiclass * loss_multiclass + mmd_weight * loss_mmd
        
        total_loss.backward()
        optimizer.step()
        
        # 统计
        batch_size = images_40x.size(0)
        running_loss += total_loss.item() * batch_size
        running_loss_binary += loss_binary.item() * batch_size
        if use_multiclass:
            running_loss_multiclass += loss_multiclass.item() * batch_size
        running_loss_mmd += loss_mmd.item() * batch_size
        
        # 准确率
        _, pred_binary = torch.max(outputs['logits_final_binary'], 1)
        correct_binary += (pred_binary == labels_binary).sum().item()
        
        if use_multiclass and outputs['logits_final_multiclass'] is not None:
            _, pred_multiclass = torch.max(outputs['logits_final_multiclass'], 1)
            correct_multiclass += (pred_multiclass == labels_multiclass).sum().item()
        
        total_samples += batch_size
    
    epoch_loss = running_loss / total_samples
    epoch_loss_binary = running_loss_binary / total_samples
    epoch_loss_multiclass = running_loss_multiclass / total_samples if use_multiclass else 0.0
    epoch_loss_mmd = running_loss_mmd / total_samples
    epoch_acc_binary = 100.0 * correct_binary / total_samples
    epoch_acc_multiclass = 100.0 * correct_multiclass / total_samples if use_multiclass else 0.0
    
    return {
        'loss': epoch_loss,
        'loss_binary': epoch_loss_binary,
        'loss_multiclass': epoch_loss_multiclass,
        'loss_mmd': epoch_loss_mmd,
        'acc_binary': epoch_acc_binary,
        'acc_multiclass': epoch_acc_multiclass,
        'mmd_weight': mmd_weight
    }


def validate(model, val_loader, criterion_binary, criterion_multiclass, criterion_mmd,
            device, alpha_multiclass, use_multiclass):
    """验证模型"""
    model.eval()
    
    running_loss = 0.0
    running_loss_binary = 0.0
    running_loss_multiclass = 0.0
    running_loss_mmd = 0.0
    
    correct_binary = 0
    correct_multiclass = 0
    total_samples = 0
    
    with torch.no_grad():
        for images_40x, images_100x, labels_binary, labels_multiclass, _ in val_loader:
            images_40x = images_40x.to(device)
            images_100x = images_100x.to(device)
            labels_binary = labels_binary.to(device)
            labels_multiclass = labels_multiclass.to(device)
            
            outputs = model(images_40x, images_100x)
            
            loss_binary = criterion_binary(outputs['logits_final_binary'], labels_binary)
            
            loss_multiclass = torch.tensor(0.0).to(device)
            if use_multiclass and outputs['logits_final_multiclass'] is not None:
                loss_multiclass = criterion_multiclass(outputs['logits_final_multiclass'], labels_multiclass)
            
            loss_mmd = criterion_mmd(outputs['features_40x'], outputs['features_100x'])
            total_loss = loss_binary + alpha_multiclass * loss_multiclass + loss_mmd
            
            batch_size = images_40x.size(0)
            running_loss += total_loss.item() * batch_size
            running_loss_binary += loss_binary.item() * batch_size
            if use_multiclass:
                running_loss_multiclass += loss_multiclass.item() * batch_size
            running_loss_mmd += loss_mmd.item() * batch_size
            
            _, pred_binary = torch.max(outputs['logits_final_binary'], 1)
            correct_binary += (pred_binary == labels_binary).sum().item()
            
            if use_multiclass and outputs['logits_final_multiclass'] is not None:
                _, pred_multiclass = torch.max(outputs['logits_final_multiclass'], 1)
                correct_multiclass += (pred_multiclass == labels_multiclass).sum().item()
            
            total_samples += batch_size
    
    epoch_loss = running_loss / total_samples
    epoch_loss_binary = running_loss_binary / total_samples
    epoch_loss_multiclass = running_loss_multiclass / total_samples if use_multiclass else 0.0
    epoch_loss_mmd = running_loss_mmd / total_samples
    epoch_acc_binary = 100.0 * correct_binary / total_samples
    epoch_acc_multiclass = 100.0 * correct_multiclass / total_samples if use_multiclass else 0.0
    
    return {
        'loss': epoch_loss,
        'loss_binary': epoch_loss_binary,
        'loss_multiclass': epoch_loss_multiclass,
        'loss_mmd': epoch_loss_mmd,
        'acc_binary': epoch_acc_binary,
        'acc_multiclass': epoch_acc_multiclass
    }


def main(args):
    # 加载配置
    config = load_config(args.config)
    
    print(f"\n{'='*80}")
    print(f"实验: {config['experiment_name']}")
    print(f"配置: {config.get('stream_config', 'xception_swin')}")
    print(f"{'='*80}\n")
    
    # 设置随机种子
    set_seed(config.get('seed', 42))
    
    # 设置设备
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    checkpoint_dir = config['checkpoint']['save_dir']
    results_dir = config['results']['save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存配置
    config_save_path = os.path.join(results_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    
    # 创建数据集
    data_cfg = config['data']
    image_size_40x = data_cfg.get('image_size_40x', 299)  # Xception需要299
    image_size_100x = data_cfg.get('image_size_100x', 224 if config.get('stream_config') == 'xception_swin' else 299)
    
    print("\n加载数据集...")
    train_dataset = PairedMagnificationDataset(
        csv_40x=data_cfg['train_csv_40x'],
        csv_100x=data_cfg['train_csv_100x'],
        transform_40x=get_transforms(image_size=image_size_40x, is_training=True),
        transform_100x=get_transforms(image_size=image_size_100x, is_training=True)
    )
    
    val_dataset = PairedMagnificationDataset(
        csv_40x=data_cfg['val_csv_40x'],
        csv_100x=data_cfg['val_csv_100x'],
        transform_40x=get_transforms(image_size=image_size_40x, is_training=False),
        transform_100x=get_transforms(image_size=image_size_100x, is_training=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get('batch_size', 16),
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get('batch_size', 16),
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    print("\n创建模型...")
    model = get_daf_mmd_xception_model(
        stream_config=config.get('stream_config', 'xception_swin'),
        num_classes_binary=config['model']['num_classes_binary'],
        num_classes_multiclass=config['model']['num_classes_multiclass'],
        use_multiclass=config['model'].get('use_multiclass', True),
        use_pretrained=config['model'].get('pretrained', True)
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建损失函数
    train_cfg = config['training']
    use_multiclass = config['model'].get('use_multiclass', True)
    
    if train_cfg.get('loss') == 'focal_loss':
        alpha_value = train_cfg.get('focal_alpha', None)
        if isinstance(alpha_value, (int, float)):
            alpha_value = None
            
        criterion_binary = FocalLoss(
            alpha=alpha_value,
            gamma=train_cfg.get('focal_gamma', 2.0),
            reduction='mean'
        )
        criterion_multiclass = FocalLoss(
            alpha=alpha_value,
            gamma=train_cfg.get('focal_gamma', 2.0),
            reduction='mean'
        ) if use_multiclass else None
    else:
        criterion_binary = nn.CrossEntropyLoss()
        criterion_multiclass = nn.CrossEntropyLoss() if use_multiclass else None
    
    criterion_mmd = MMDLoss(sigma=train_cfg.get('mmd_sigma', 1.0))
    alpha_multiclass = train_cfg.get('alpha_multiclass', 1.0)
    
    # 创建优化器
    optimizer_name = train_cfg.get('optimizer', 'adam').lower()
    lr = train_cfg.get('learning_rate', 0.0001)
    weight_decay = train_cfg.get('weight_decay', 0.0001)
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler_name = train_cfg.get('scheduler', 'cosine')
    epochs = train_cfg.get('epochs', 100)
    
    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None
    
    # 训练循环
    print(f"\n开始训练 ({epochs} epochs)...")
    print("⚠️  注意: 早停和模型保存以八分类准确率为准！")
    
    best_val_acc_multiclass = 0.0  # 以八分类为准
    best_val_acc_binary = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    patience = train_cfg.get('patience', 15) if train_cfg.get('early_stopping', True) else epochs
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc_binary': [],
        'val_acc_binary': [],
        'train_acc_multiclass': [],
        'val_acc_multiclass': [],
        'train_loss_mmd': [],
        'val_loss_mmd': [],
        'lr': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print('-' * 80)
        
        train_metrics = train_one_epoch(
            model, train_loader, criterion_binary, criterion_multiclass, criterion_mmd,
            optimizer, device, epoch, epochs, alpha_multiclass,
            train_cfg.get('mmd_initial_weight', 0.05),
            train_cfg.get('mmd_final_weight', 1.0),
            train_cfg.get('mmd_start_ratio', 0.5),
            use_multiclass
        )
        
        val_metrics = validate(
            model, val_loader, criterion_binary, criterion_multiclass, criterion_mmd,
            device, alpha_multiclass, use_multiclass
        )
        
        if scheduler is not None:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc_binary'].append(train_metrics['acc_binary'])
        history['val_acc_binary'].append(val_metrics['acc_binary'])
        history['train_acc_multiclass'].append(train_metrics['acc_multiclass'])
        history['val_acc_multiclass'].append(val_metrics['acc_multiclass'])
        history['train_loss_mmd'].append(train_metrics['loss_mmd'])
        history['val_loss_mmd'].append(val_metrics['loss_mmd'])
        history['lr'].append(current_lr)
        
        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"Train Binary Acc: {train_metrics['acc_binary']:.2f}% | Val Binary Acc: {val_metrics['acc_binary']:.2f}%")
        if use_multiclass:
            print(f"Train 8-class Acc: {train_metrics['acc_multiclass']:.2f}% | Val 8-class Acc: {val_metrics['acc_multiclass']:.2f}% ⭐")
        print(f"MMD Weight: {train_metrics['mmd_weight']:.4f} | MMD Loss: {train_metrics['loss_mmd']:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # 保存最佳模型 - 以八分类准确率为准
        if use_multiclass:
            current_val_acc = val_metrics['acc_multiclass']
            is_best = current_val_acc > best_val_acc_multiclass
        else:
            current_val_acc = val_metrics['acc_binary']
            is_best = current_val_acc > best_val_acc_binary
        
        if is_best:
            if use_multiclass:
                best_val_acc_multiclass = current_val_acc
            
            best_val_acc_binary = val_metrics['acc_binary']
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_multiclass': best_val_acc_multiclass if use_multiclass else 0.0,
                'val_acc_binary': val_metrics['acc_binary'],
                'val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)
            print(f"✓ 保存最佳模型 (8-class Acc: {current_val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping - 以八分类为准
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (基于8-class accuracy)")
            break
    
    total_time = time.time() - start_time
    history['total_training_time_seconds'] = total_time
    history['best_val_acc_multiclass'] = best_val_acc_multiclass
    
    print(f"\n{'='*80}")
    print(f"训练完成!")
    print(f"总时间: {total_time/60:.2f} 分钟")
    print(f"最佳验证八分类准确率: {best_val_acc_multiclass:.2f}% ⭐")
    print(f"最佳验证二分类准确率: {best_val_acc_binary:.2f}%")
    print(f"{'='*80}\n")
    
    # 保存最后一个epoch的模型作为备用
    last_model_path = os.path.join(checkpoint_dir, 'last_model.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc_multiclass': val_metrics['acc_multiclass'],
        'val_acc_binary': val_metrics['acc_binary'],
        'val_loss': val_metrics['loss'],
        'config': config
    }, last_model_path)

    # 确保 best_model.pth 存在
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if not os.path.exists(best_model_path):
        print(f"⚠️  警告: 未找到 best_model.pth，将使用最后一个 epoch 的模型作为最佳模型。")
        import shutil
        shutil.copy(last_model_path, best_model_path)
    
    # 保存训练历史
    history_path = os.path.join(results_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线
    plot_path = os.path.join(results_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    print("✓ 训练历史已保存")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DAF-MMD Net with Xception 训练脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()
    
    main(args)


