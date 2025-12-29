"""
CTransPath模型定义 (Groups 2-7)
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
    来自TransPath项目: https://github.com/Xiyue-Wang/TransPath
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, **kwargs):
        super().__init__()
        # **kwargs 用于接收timm可能传入的其他参数（如strict_img_size）

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


class CTransPath(nn.Module):
    """
    CTransPath模型包装器
    支持加载预训练权重和自定义分类头
    """
    def __init__(self, num_classes=8, pretrained_path=None, 
                 classifier_type='linear', freeze_backbone=True):
        """
        Args:
            num_classes: 分类类别数
            pretrained_path: 预训练权重路径 (ctranspath.pth 或 None使用ImageNet权重)
            classifier_type: 分类头类型 ('linear' 或 'mlp')
            freeze_backbone: 是否冻结backbone
        """
        super(CTransPath, self).__init__()
        
        # 加载Swin Transformer作为backbone
        # 根据预训练权重选择对应的模型架构
        if pretrained_path is not None and pretrained_path != 'imagenet':
            # CTransPath使用的配置（参考TransPath/ctran.py）
            print("使用 CTransPath 架构（Swin-Tiny + ConvStem）")
            # 直接使用swin_tiny_patch4_window7_224并替换embed_layer
            # swin_tiny默认: embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24]
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
            
            # CTransPath权重格式: {'model': state_dict}
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 移除分类头的权重（如果存在）
            state_dict = {k: v for k, v in state_dict.items() 
                         if not k.startswith('head')}
            
            # 手动加载权重，跳过维度不匹配的层
            model_dict = self.backbone.state_dict()
            loaded_count = 0
            skipped_count = 0
            
            for key, value in state_dict.items():
                if key in model_dict:
                    if model_dict[key].shape == value.shape:
                        model_dict[key] = value
                        loaded_count += 1
                    else:
                        # 维度不匹配，跳过
                        skipped_count += 1
                        # print(f"  跳过 {key}: shape {value.shape} vs {model_dict[key].shape}")
            
            # 加载到模型
            self.backbone.load_state_dict(model_dict)
            
            print(f"✓ CTransPath权重部分加载完成")
            print(f"  成功加载: {loaded_count} 个参数")
            print(f"  跳过不匹配: {skipped_count} 个参数")
            print(f"  注意: 由于timm版本不兼容，部分层使用随机初始化")
        elif pretrained_path == 'imagenet':
            # 加载ImageNet预训练权重
            imagenet_weight_path = 'swin_base_imagenet.pth'
            if os.path.exists(imagenet_weight_path):
                print(f"加载ImageNet预训练权重: {imagenet_weight_path}")
                state_dict = torch.load(imagenet_weight_path, map_location='cpu')
                
                # 处理可能的key不匹配
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # 移除分类头的权重
                state_dict = {k: v for k, v in state_dict.items() 
                             if not k.startswith('head')}
                
                missing_keys, unexpected_keys = self.backbone.load_state_dict(
                    state_dict, strict=False
                )
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")
                print("✓ ImageNet权重加载完成")
            else:
                print(f"⚠️  警告：未找到ImageNet权重文件 {imagenet_weight_path}")
                print("    将使用随机初始化的权重")
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
            print("使用Linear分类头")
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        elif classifier_type == 'mlp':
            print("使用MLP分类头")
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")
        
        # 初始化分类头权重
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """初始化分类头权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 (batch_size, 3, H, W)
        
        Returns:
            输出logits (batch_size, num_classes)
        """
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_feature_dim(self):
        """返回特征维度"""
        return self.feature_dim


def get_ctranspath_model(num_classes=8, group='group2', 
                         ctranspath_weight_path='ctranspath.pth'):
    """
    根据实验组配置创建CTransPath模型
    
    Args:
        num_classes: 分类类别数
        group: 实验组名称 ('group2'-'group7')
        ctranspath_weight_path: CTransPath预训练权重路径
    
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
    print(f"创建模型: {group}")
    print(f"配置: {config}")
    print(f"{'='*50}\n")
    
    model = CTransPath(
        num_classes=num_classes,
        pretrained_path=config['pretrained_path'],
        classifier_type=config['classifier_type'],
        freeze_backbone=config['freeze_backbone']
    )
    
    return model


if __name__ == "__main__":
    # 测试模型
    print("测试Group 2模型 (CTransPath + Linear + Frozen):")
    model = get_ctranspath_model(num_classes=8, group='group2')
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

