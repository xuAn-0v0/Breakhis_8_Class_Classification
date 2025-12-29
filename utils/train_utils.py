"""
训练相关工具函数
"""
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm
import os


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    训练一个epoch
    
    Args:
        model: 模型
        dataloader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
    
    Returns:
        平均损失和准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, phase='Val'):
    """
    验证/测试模型
    
    Args:
        model: 模型
        dataloader: 验证/测试数据加载器
        criterion: 损失函数
        device: 设备
        phase: 阶段名称
    
    Returns:
        平均损失、准确率、所有预测和真实标签
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"[{phase}]")
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 保存预测结果
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, save_path):
    """
    保存模型checkpoint
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        best_acc: 最佳准确率
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc
    }
    
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved to {save_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device='cpu'):
    """
    加载模型checkpoint
    
    Args:
        model: 模型
        checkpoint_path: checkpoint路径
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
    
    Returns:
        起始epoch和最佳准确率
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_acc = checkpoint.get('best_acc', 0.0)
    
    print(f"✓ Checkpoint loaded from {checkpoint_path}")
    print(f"  Starting from epoch {start_epoch}, best acc: {best_acc:.2f}%")
    
    return start_epoch, best_acc


def get_optimizer(model, optimizer_name='adam', lr=0.001, weight_decay=1e-4, momentum=0.9):
    """
    获取优化器
    
    Args:
        model: 模型
        optimizer_name: 优化器名称 ('adam', 'sgd', 'adamw')
        lr: 学习率
        weight_decay: 权重衰减
        momentum: 动量（仅用于SGD）
    
    Returns:
        优化器
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def get_scheduler(optimizer, scheduler_name='cosine', epochs=100, step_size=30, gamma=0.1):
    """
    获取学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_name: 调度器名称 ('cosine', 'step', 'plateau', None)
        epochs: 总epoch数（用于cosine）
        step_size: 步长（用于step）
        gamma: 衰减率
    
    Returns:
        学习率调度器
    """
    if scheduler_name is None or scheduler_name.lower() == 'none':
        return None
    
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'step':
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=gamma, patience=10)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler

