"""
CTransPath 多任务学习版本 (Groups 2-7)
支持同时进行 2分类（良性/恶性）和 8分类（具体类别）
"""
import torch
import torch.nn as nn
import timm
import os

# 兼容新旧版本的timm
try:
    from timm.layers import to_2tuple
except ImportError:
    try:
        from timm.models.layers import to_2tuple
    except ImportError:
        from timm.models.layers.helpers import to_2tuple


class ConvStem(nn.Module):
    """
    CTransPath使用的卷积Stem层
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, **kwargs):
        super().__init__()
        
        assert patch_size == 4
        assert embed_dim % 8 == 0
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class CTransPathMultiTask(nn.Module):
    """
    CTransPath模型包装器，支持多任务学习
    """
    def __init__(self, num_classes_binary=2, num_classes_multiclass=8,
                 pretrained_path=None, classifier_type='linear', 
                 freeze_backbone=True, use_multiclass=True):
        """
        Args:
            num_classes_binary: 二分类类别数（默认2：良性/恶性）
            num_classes_multiclass: 八分类类别数（默认8）
            pretrained_path: 预训练权重路径
            classifier_type: 分类头类型 ('linear' 或 'mlp')
            freeze_backbone: 是否冻结backbone
            use_multiclass: 是否启用八分类任务
        """
        super(CTransPathMultiTask, self).__init__()
        
        self.num_classes_binary = num_classes_binary
        self.num_classes_multiclass = num_classes_multiclass
        self.use_multiclass = use_multiclass
        
        # 加载Swin Transformer作为backbone
        if pretrained_path is not None and pretrained_path != 'imagenet':
            # CTransPath使用的配置
            print("使用 CTransPath 架构（Swin-Tiny + ConvStem）")
            self.backbone = timm.create_model(
                'swin_tiny_patch4_window7_224',
                embed_layer=ConvStem,
                pretrained=False,
                num_classes=0
            )
        else:
            # ImageNet使用标准Swin-Base
            print("使用 Swin-Base 架构（ImageNet）")
            self.backbone = timm.create_model(
                'swin_base_patch4_window7_224',
                pretrained=False,
                num_classes=0
            )
        
        # 获取backbone输出维度
        self.feature_dim = self.backbone.num_features
        
        # 加载预训练权重
        if pretrained_path is not None and pretrained_path != 'imagenet':
            print(f"加载CTransPath预训练权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 移除分类头的权重
            state_dict = {k: v for k, v in state_dict.items() 
                         if not k.startswith('head')}
            
            # 手动加载权重
            model_dict = self.backbone.state_dict()
            loaded_count = 0
            skipped_count = 0
            
            for key, value in state_dict.items():
                if key in model_dict:
                    if model_dict[key].shape == value.shape:
                        model_dict[key] = value
                        loaded_count += 1
                    else:
                        skipped_count += 1
            
            self.backbone.load_state_dict(model_dict)
            
            print(f"✓ CTransPath权重部分加载完成")
            print(f"  成功加载: {loaded_count} 个参数")
            print(f"  跳过不匹配: {skipped_count} 个参数")
        elif pretrained_path == 'imagenet':
            # 加载ImageNet预训练权重
            imagenet_weight_path = 'swin_base_imagenet.pth'
            if os.path.exists(imagenet_weight_path):
                print(f"加载ImageNet预训练权重: {imagenet_weight_path}")
                state_dict = torch.load(imagenet_weight_path, map_location='cpu')
                
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                state_dict = {k: v for k, v in state_dict.items() 
                             if not k.startswith('head')}
                
                missing_keys, unexpected_keys = self.backbone.load_state_dict(
                    state_dict, strict=False
                )
                print("✓ ImageNet权重加载完成")
            else:
                print(f"⚠️  警告：未找到ImageNet权重文件")
        else:
            print("使用随机初始化的权重")
        
        # 冻结backbone
        if freeze_backbone:
            print("冻结backbone参数")
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            print("backbone参数可训练（微调模式）")
        
        # 创建分类头
        if classifier_type == 'linear':
            print("使用Linear分类头（多任务）")
            self.classifier_binary = nn.Linear(self.feature_dim, num_classes_binary)
            if use_multiclass:
                self.classifier_multiclass = nn.Linear(self.feature_dim, num_classes_multiclass)
        elif classifier_type == 'mlp':
            print("使用MLP分类头（多任务）")
            # 共享的中间层
            self.shared_mlp = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
            # 二分类头
            self.classifier_binary = nn.Linear(256, num_classes_binary)
            # 八分类头
            if use_multiclass:
                self.classifier_multiclass = nn.Linear(256, num_classes_multiclass)
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")
        
        self.classifier_type = classifier_type
        
        # 初始化分类头权重
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """初始化分类头权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in self.backbone.modules():
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 (batch_size, 3, H, W)
        
        Returns:
            字典包含两个分类任务的输出:
            {
                'logits_binary': (batch_size, num_classes_binary),
                'logits_multiclass': (batch_size, num_classes_multiclass) 或 None
            }
        """
        # 共享特征提取
        features = self.backbone(x)
        
        # 根据分类头类型处理
        if self.classifier_type == 'mlp':
            features = self.shared_mlp(features)
        
        # 二分类
        logits_binary = self.classifier_binary(features)
        
        # 八分类（如果启用）
        logits_multiclass = None
        if self.use_multiclass:
            logits_multiclass = self.classifier_multiclass(features)
        
        return {
            'logits_binary': logits_binary,
            'logits_multiclass': logits_multiclass
        }
    
    def get_feature_dim(self):
        """返回特征维度"""
        return self.feature_dim


def get_ctranspath_multitask_model(num_classes_binary=2, num_classes_multiclass=8,
                                   group='group2', ctranspath_weight_path='ctranspath.pth',
                                   use_multiclass=True):
    """
    根据实验组配置创建CTransPath多任务模型
    
    Args:
        num_classes_binary: 二分类类别数
        num_classes_multiclass: 八分类类别数
        group: 实验组名称 ('group2'-'group7')
        ctranspath_weight_path: CTransPath预训练权重路径
        use_multiclass: 是否启用八分类任务
    
    Returns:
        模型实例
    """
    group_configs = {
        # Groups 2-4: CTransPath预训练权重
        'group2': {
            'pretrained_path': ctranspath_weight_path,
            'classifier_type': 'linear',
            'freeze_backbone': True
        },
        'group3': {
            'pretrained_path': ctranspath_weight_path,
            'classifier_type': 'mlp',
            'freeze_backbone': True
        },
        'group4': {
            'pretrained_path': ctranspath_weight_path,
            'classifier_type': 'linear',
            'freeze_backbone': False
        },
        # Groups 5-7: ImageNet预训练权重
        'group5': {
            'pretrained_path': None,
            'classifier_type': 'linear',
            'freeze_backbone': True
        },
        'group6': {
            'pretrained_path': None,
            'classifier_type': 'mlp',
            'freeze_backbone': True
        },
        'group7': {
            'pretrained_path': None,
            'classifier_type': 'linear',
            'freeze_backbone': False
        },
    }
    
    if group not in group_configs:
        raise ValueError(f"Unknown group: {group}")
    
    config = group_configs[group]
    print(f"\n{'='*50}")
    print(f"创建多任务模型: {group}")
    print(f"配置: {config}")
    print(f"{'='*50}\n")
    
    model = CTransPathMultiTask(
        num_classes_binary=num_classes_binary,
        num_classes_multiclass=num_classes_multiclass,
        pretrained_path=config['pretrained_path'],
        classifier_type=config['classifier_type'],
        freeze_backbone=config['freeze_backbone'],
        use_multiclass=use_multiclass
    )
    
    return model


if __name__ == "__main__":
    # 测试模型
    print("测试 Group 2 多任务模型 (CTransPath + Linear + Frozen):")
    model = get_ctranspath_multitask_model(
        num_classes_binary=2,
        num_classes_multiclass=8,
        group='group2',
        use_multiclass=True
    )
    
    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Binary output shape: {outputs['logits_binary'].shape}")
    print(f"Multiclass output shape: {outputs['logits_multiclass'].shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
