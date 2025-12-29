"""
CTransPath模型定义 - 使用TransPath原始实现
直接从TransPath项目复制，避免timm版本兼容问题
"""
import torch
import torch.nn as nn
import sys
import os

# 添加TransPath到路径
transpath_dir = os.path.join(os.path.dirname(__file__), '../TransPath')
if os.path.exists(transpath_dir):
    sys.path.insert(0, transpath_dir)

try:
    from ctran import ctranspath, ConvStem
    TRANSPATH_AVAILABLE = True
except ImportError:
    print("⚠️  警告: 无法导入TransPath原始实现")
    TRANSPATH_AVAILABLE = False
    ConvStem = None


def get_ctranspath_legacy(num_classes=8, pretrained_path='ctranspath.pth', 
                          classifier_type='linear', freeze_backbone=True, dropout=0.0):
    """
    使用TransPath原始实现加载CTransPath模型
    
    Args:
        num_classes: 分类数
        pretrained_path: 预训练权重路径
        classifier_type: 分类头类型
        freeze_backbone: 是否冻结
        dropout: Dropout率（仅用于linear类型）
    
    Returns:
        完整模型
    """
    if not TRANSPATH_AVAILABLE:
        raise ImportError("TransPath模块不可用，请检查TransPath目录")
    
    # 创建模型
    print("使用TransPath原始实现创建模型")
    model = ctranspath()
    
    # 加载预训练权重
    if os.path.exists(pretrained_path):
        print(f"加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        print("✓ 权重加载成功")
    
    # 冻结backbone
    if freeze_backbone:
        print("冻结backbone参数")
        for name, param in model.named_parameters():
            if not name.startswith('head'):
                param.requires_grad = False
    
    # 修改分类头
    feature_dim = model.head.in_features if hasattr(model.head, 'in_features') else 768
    
    if classifier_type == 'linear':
        if dropout > 0:
            print(f"使用Linear分类头 + Dropout({dropout}) ({feature_dim} → {num_classes})")
            model.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feature_dim, num_classes)
            )
        else:
            print(f"使用Linear分类头 ({feature_dim} → {num_classes})")
            model.head = nn.Linear(feature_dim, num_classes)
    elif classifier_type == 'mlp':
        print(f"使用MLP分类头 ({feature_dim} → ... → {num_classes})")
        model.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    return model


if __name__ == "__main__":
    # 测试
    model = get_ctranspath_legacy(
        num_classes=8,
        pretrained_path='ctranspath.pth',
        classifier_type='linear',
        freeze_backbone=True
    )
    
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input: {x.shape}, Output: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
