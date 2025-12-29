"""
通用多任务学习训练脚本
支持 Group1 CNN 和 Group2-4 CTransPath 模型，以及 timm 库模型
同时训练2分类（良性/恶性）和 8分类（具体类别）
"""
import os
import sys
import argparse
import yaml
import json
import time
from datetime import datetime
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__)) if os.path.dirname(os.path.abspath(__file__)) else '/songjian/zixuan'
sys.path.insert(0, script_dir)  # 添加主目录

from data.multitask_dataset import MultiTaskDataset
from utils.focal_loss import FocalLoss
from utils.metrics import plot_confusion_matrix, plot_training_history

# 导入模型
from models.simple_cnn_multitask import SimpleCNNMultiTask
from models.ctranspath_multitask import get_ctranspath_multitask_model
from models.timm_multitask import get_timm_multitask_model


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(image_size=224, mode='train', use_enhanced_aug=False):
    """获取数据增强"""
    if mode == 'train':
        if use_enhanced_aug:
             return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
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
            group=config.get('group', 'ctranspath'),
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


def train_one_epoch(model, dataloader, criterion_binary, criterion_multiclass,
                   optimizer, device, alpha_multiclass, use_multiclass):
    """训练一个epoch"""
    model.train()
    
    running_loss = 0.0
    running_loss_binary = 0.0
    running_loss_multiclass = 0.0
    
    correct_binary = 0
    correct_multiclass = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for images, labels_binary, labels_multiclass in progress_bar:
        images = images.to(device)
        labels_binary = labels_binary.to(device)
        labels_multiclass = labels_multiclass.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        
        # 计算二分类损失
        loss_binary = criterion_binary(outputs['logits_binary'], labels_binary)
        
        # 计算八分类损失（如果启用）
        loss_multiclass = torch.tensor(0.0).to(device)
        if use_multiclass and outputs['logits_multiclass'] is not None:
            loss_multiclass = criterion_multiclass(outputs['logits_multiclass'], labels_multiclass)
        
        # 总损失
        total_loss = loss_binary + alpha_multiclass * loss_multiclass
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        # 统计
        batch_size = images.size(0)
        running_loss += total_loss.item() * batch_size
        running_loss_binary += loss_binary.item() * batch_size
        if use_multiclass:
            running_loss_multiclass += loss_multiclass.item() * batch_size
        
        # 计算准确率
        _, pred_binary = torch.max(outputs['logits_binary'], 1)
        correct_binary += (pred_binary == labels_binary).sum().item()
        
        if use_multiclass and outputs['logits_multiclass'] is not None:
            _, pred_multiclass = torch.max(outputs['logits_multiclass'], 1)
            correct_multiclass += (pred_multiclass == labels_multiclass).sum().item()
        
        total_samples += batch_size
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'acc_bin': f'{100.0 * correct_binary / total_samples:.2f}%',
            'acc_multi': f'{100.0 * correct_multiclass / total_samples:.2f}%' if use_multiclass else 'N/A'
        })
    
    # 计算平均指标
    epoch_loss = running_loss / total_samples
    epoch_loss_binary = running_loss_binary / total_samples
    epoch_loss_multiclass = running_loss_multiclass / total_samples if use_multiclass else 0.0
    epoch_acc_binary = 100.0 * correct_binary / total_samples
    epoch_acc_multiclass = 100.0 * correct_multiclass / total_samples if use_multiclass else 0.0
    
    return {
        'loss': epoch_loss,
        'loss_binary': epoch_loss_binary,
        'loss_multiclass': epoch_loss_multiclass,
        'acc_binary': epoch_acc_binary,
        'acc_multiclass': epoch_acc_multiclass
    }


def validate(model, dataloader, criterion_binary, criterion_multiclass,
            device, alpha_multiclass, use_multiclass):
    """验证模型"""
    model.eval()
    
    running_loss = 0.0
    running_loss_binary = 0.0
    running_loss_multiclass = 0.0
    
    correct_binary = 0
    correct_multiclass = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels_binary, labels_multiclass in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            labels_binary = labels_binary.to(device)
            labels_multiclass = labels_multiclass.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss_binary = criterion_binary(outputs['logits_binary'], labels_binary)
            
            loss_multiclass = torch.tensor(0.0).to(device)
            if use_multiclass and outputs['logits_multiclass'] is not None:
                loss_multiclass = criterion_multiclass(outputs['logits_multiclass'], labels_multiclass)
            
            total_loss = loss_binary + alpha_multiclass * loss_multiclass
            
            # 统计
            batch_size = images.size(0)
            running_loss += total_loss.item() * batch_size
            running_loss_binary += loss_binary.item() * batch_size
            if use_multiclass:
                running_loss_multiclass += loss_multiclass.item() * batch_size
            
            # 计算准确率
            _, pred_binary = torch.max(outputs['logits_binary'], 1)
            correct_binary += (pred_binary == labels_binary).sum().item()
            
            if use_multiclass and outputs['logits_multiclass'] is not None:
                _, pred_multiclass = torch.max(outputs['logits_multiclass'], 1)
                correct_multiclass += (pred_multiclass == labels_multiclass).sum().item()
            
            total_samples += batch_size
    
    # 计算平均指标
    epoch_loss = running_loss / total_samples
    epoch_loss_binary = running_loss_binary / total_samples
    epoch_loss_multiclass = running_loss_multiclass / total_samples if use_multiclass else 0.0
    epoch_acc_binary = 100.0 * correct_binary / total_samples
    epoch_acc_multiclass = 100.0 * correct_multiclass / total_samples if use_multiclass else 0.0
    
    return {
        'loss': epoch_loss,
        'loss_binary': epoch_loss_binary,
        'loss_multiclass': epoch_loss_multiclass,
        'acc_binary': epoch_acc_binary,
        'acc_multiclass': epoch_acc_multiclass
    }


def freeze_backbone(model, freeze=True):
    """冻结/解冻backbone参数"""
    # 针对 TimmMultiTaskModel (backbone属性)
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = not freeze
        print(f"{'冻结' if freeze else '解冻'} Backbone 参数")
    # 针对 SimpleCNNMultiTask (features属性)
    elif hasattr(model, 'features'):
        for param in model.features.parameters():
            param.requires_grad = not freeze
        print(f"{'冻结' if freeze else '解冻'} Features 参数")
    else:
        print("未找到可冻结的 backbone/features")

def main(args):
    """主训练函数"""
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n{'='*80}")
    print(f"实验: {config['experiment_name']}")
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
    image_size = data_cfg.get('image_size', 224)
    use_enhanced_aug = data_cfg.get('use_enhanced_aug', False)
    
    print("\n加载数据集...")
    train_dataset = MultiTaskDataset(
        csv_file=data_cfg['train_csv'],
        root_dir=data_cfg['dataset_root'],
        transform=get_transforms(image_size, mode='train', use_enhanced_aug=use_enhanced_aug)
    )
    
    val_dataset = MultiTaskDataset(
        csv_file=data_cfg['val_csv'],
        root_dir=data_cfg['dataset_root'],
        transform=get_transforms(image_size, mode='val')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get('batch_size', 32),
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get('batch_size', 32),
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    print("\n创建模型...")
    model = create_model(config)
    model = model.to(device)
    
    # 处理 Backbone 冻结
    freeze_epochs = config['model'].get('freeze_backbone_epochs', 0)
    if freeze_epochs > 0:
        freeze_backbone(model, freeze=True)
    
    # 统计参数
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
        if use_multiclass:
            criterion_multiclass = FocalLoss(
                alpha=alpha_value,
                gamma=train_cfg.get('focal_gamma', 2.0),
                reduction='mean'
            )
        else:
            criterion_multiclass = None
    else:
        criterion_binary = nn.CrossEntropyLoss()
        criterion_multiclass = nn.CrossEntropyLoss() if use_multiclass else None
    
    alpha_multiclass = train_cfg.get('alpha_multiclass', 1.0)
    
    # 创建优化器
    optimizer_name = train_cfg.get('optimizer', 'adam').lower()
    lr = train_cfg.get('learning_rate', 0.001)
    weight_decay = train_cfg.get('weight_decay', 0.0001)
    
    def get_optimizer(parameters):
        if optimizer_name == 'adam':
            return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()))

    # 创建学习率调度器
    scheduler_name = train_cfg.get('scheduler', 'cosine')
    epochs = train_cfg.get('epochs', 100)
    warmup_epochs = train_cfg.get('warmup_epochs', 0)
    
    scheduler = None
    if scheduler_name == 'cosine':
        # 如果有 warmup，这里只负责 warmup 之后的 cosine 部分
        # 具体实现可以用 SequentialLR，或者简单地在 warmup 后再创建 scheduler
        # 这里为了简单，如果设置了 warmup，我们会在循环中手动处理
        if warmup_epochs == 0:
             scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
             # 创建一个临时的 scheduler 占位
             scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
             
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 训练循环
    print(f"\n开始训练 ({epochs} epochs)...")
    
    best_val_loss = float('inf')
    best_val_acc_multiclass = 0.0  # 改为监控八分类
    best_val_acc_binary = 0.0      # 添加二分类初始化
    patience_counter = 0
    patience = train_cfg.get('patience', 15) if train_cfg.get('early_stopping', True) else epochs
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc_binary': [],
        'val_acc_binary': [],
        'train_acc_multiclass': [],
        'val_acc_multiclass': [],
        'lr': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print('-' * 80)
        
        # 处理 Backbone 解冻
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            print("\n>>> 解冻 Backbone 参数 <<<")
            freeze_backbone(model, freeze=False)
            # 重建优化器以包含新解冻的参数
            # 注意：这会重置动量等状态，对于 fine-tuning 是可接受的，或者可以将 learning rate 调小
            optimizer = get_optimizer(model.parameters())
            # 重置 scheduler
            if scheduler_name == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - epoch + 1)
        
        # 手动 Warmup
        if epoch <= warmup_epochs:
            warmup_lr = lr * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Warmup: LR set to {warmup_lr:.6f}")

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion_binary, criterion_multiclass,
            optimizer, device, alpha_multiclass, use_multiclass
        )
        
        # 验证
        val_metrics = validate(
            model, val_loader, criterion_binary, criterion_multiclass,
            device, alpha_multiclass, use_multiclass
        )
        
        # 更新学习率 (Warmup 结束后)
        if epoch > warmup_epochs and scheduler is not None:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc_binary'].append(train_metrics['acc_binary'])
        history['val_acc_binary'].append(val_metrics['acc_binary'])
        history['train_acc_multiclass'].append(train_metrics['acc_multiclass'])
        history['val_acc_multiclass'].append(val_metrics['acc_multiclass'])
        history['lr'].append(current_lr)
        
        # 打印结果
        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"Train Binary Acc: {train_metrics['acc_binary']:.2f}% | Val Binary Acc: {val_metrics['acc_binary']:.2f}%")
        if use_multiclass:
            print(f"Train 8-class Acc: {train_metrics['acc_multiclass']:.2f}% | Val 8-class Acc: {val_metrics['acc_multiclass']:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # 保存最佳模型 - 以八分类准确率为准
        if use_multiclass:
            current_val_acc = val_metrics['acc_multiclass']
            is_best = current_val_acc > best_val_acc_multiclass
        else:
            current_val_acc = val_metrics['acc_binary']
            is_best = current_val_acc > best_val_acc_binary # 降级处理

        if is_best:
            if use_multiclass:
                best_val_acc_multiclass = current_val_acc
            
            # 同时更新当前的最佳二分类准确率以便记录
            best_val_acc_binary = val_metrics['acc_binary']
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            # 保存模型
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
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
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # 训练完成
    total_time = time.time() - start_time
    history['total_training_time_seconds'] = total_time
    history['best_val_acc_multiclass'] = best_val_acc_multiclass
    history['best_val_acc_binary'] = best_val_acc_binary
    
    print(f"\n{'='*80}")
    print(f"训练完成!")
    print(f"总时间: {total_time/60:.2f} 分钟")
    if use_multiclass:
        print(f"最佳验证八分类准确率: {best_val_acc_multiclass:.2f}% ⭐")
    print(f"最佳验证二分类准确率: {best_val_acc_binary:.2f}%")
    print(f"{'='*80}\n")
    
    # 保存训练历史
    history_path = os.path.join(results_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线
    plot_path = os.path.join(results_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    print("✓ 训练历史已保存")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多任务学习训练脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()
    
    main(args)
