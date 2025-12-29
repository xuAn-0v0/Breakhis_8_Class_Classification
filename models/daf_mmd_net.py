"""
DAF-MMD Net: Dual-Stream Feature Extraction with MMD Domain Alignment
异构双流特征提取器 + MMD领域对齐 + 决策层融合
"""
import torch
import torch.nn as nn
import torchvision.models as models
import timm


class DAF_MMD_Net(nn.Module):
    """
    DAF-MMD Net模型
    
    架构：
    1. 40X流：Swin Transformer -> V_40X
    2. 100X流：Swin Transformer -> V_100X
    3. 独立分类头：FC Head_40X, FC Head_100X
    4. 元分类器：Late Fusion (P_40X, P_100X) -> P_Final
    """
    def __init__(
        self,
        num_classes_binary=2,  # 二分类：Benign vs Malignant
        num_classes_multiclass=8,  # 八分类：8个具体类别
        feature_dim=2048,  # 用于MMD对齐的统一特征维度
        swin_feature_dim=1024,  # Swin Transformer的输出特征维度
        attention_reduction=16,
        dropout=0.5,
        use_pretrained=True,
        use_multiclass=True  # 是否启用八分类
    ):
        """
        Args:
            num_classes_binary: 二分类类别数（默认2：良性/恶性）
            num_classes_multiclass: 八分类类别数（默认8：具体类别）
            feature_dim: 40X流特征维度（ResNet-152V2输出）
            swin_feature_dim: 100X流特征维度（Swin Transformer输出）
            attention_reduction: CBAM-SE注意力模块的压缩比例
            dropout: Dropout比率
            use_pretrained: 是否使用ImageNet预训练权重
            use_multiclass: 是否启用八分类任务
        """
        super(DAF_MMD_Net, self).__init__()
        
        self.num_classes_binary = num_classes_binary
        self.num_classes_multiclass = num_classes_multiclass
        self.use_multiclass = use_multiclass
        
        # ========== 40X流：Swin Transformer ==========
        # 使用Swin Transformer作为40X流的骨干网络
        swin_40x = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=use_pretrained,
            num_classes=0  # 移除分类头，返回特征
        )

        # Swin Transformer的特征提取（40X）
        self.backbone_40x = swin_40x

        # Swin Transformer的输出特征维度（40X）
        swin_40x_out_channels = self.backbone_40x.num_features

        # Swin输出的是特征向量，不需要空间注意力和GAP
        self.attention_40x = None
        self.gap_40x = None

        # 特征维度投影（统一到feature_dim用于MMD对齐）
        self.feature_proj_40x = nn.Linear(swin_40x_out_channels, feature_dim)
        
        # ========== 100X流：Swin Transformer ==========
        # 使用Swin Transformer作为100X流的骨干网络
        # timm中的swin_base_patch4_window7_224
        # 不使用features_only，直接使用标准输出（会返回特征向量）
        swin = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=use_pretrained,
            num_classes=0  # 移除分类头，返回特征
        )
        
        # Swin Transformer的特征提取（100X）
        self.backbone_100x = swin
        
        # Swin Transformer的输出特征维度
        # swin_base的输出维度通常是1024
        swin_out_channels = self.backbone_100x.num_features  # 动态获取特征维度
        
        # 由于Swin Transformer输出的是特征向量而不是特征图，
        # 不使用CBAM-SE和GAP，直接使用特征向量
        self.attention_100x = None  # Swin输出是向量，不需要空间注意力
        self.gap_100x = None  # 不需要GAP，已经是向量
        
        # 特征维度投影（统一到feature_dim，与40X流对齐，用于MMD损失）
        # 注意：这里投影到feature_dim而不是swin_feature_dim，确保MMD计算时维度一致
        self.feature_proj_100x = nn.Linear(swin_out_channels, feature_dim)
        
        # 额外的投影层用于分类（如果需要不同的维度）
        self.feature_proj_100x_classifier = nn.Linear(feature_dim, swin_feature_dim)
        
        # ========== 二分类分类头 ==========
        # 40X流的二分类头
        self.classifier_40x_binary = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes_binary)
        )
        
        # 100X流的二分类头
        self.classifier_100x_binary = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(swin_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes_binary)
        )
        
        # ========== 八分类分类头 ==========
        if use_multiclass:
            # 40X流的八分类头
            self.classifier_40x_multiclass = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, num_classes_multiclass)
            )
            
            # 100X流的八分类头
            self.classifier_100x_multiclass = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(swin_feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, num_classes_multiclass)
            )
        
        # ========== 元分类器（Late Fusion） ==========
        # 二分类元分类器
        meta_input_dim_binary = num_classes_binary * 2  # P_40X + P_100X
        self.meta_classifier_binary = nn.Sequential(
            nn.Linear(meta_input_dim_binary, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes_binary)
        )
        
        # 八分类元分类器
        if use_multiclass:
            meta_input_dim_multiclass = num_classes_multiclass * 2  # P_40X + P_100X
            self.meta_classifier_multiclass = nn.Sequential(
                nn.Linear(meta_input_dim_multiclass, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(64, num_classes_multiclass)
            )
    
    def forward(self, x_40x, x_100x):
        """
        前向传播
        
        Args:
            x_40x: 40X图像 (B, 3, H, W)
            x_100x: 100X图像 (B, 3, H, W)
        
        Returns:
            dict包含：
                二分类：
                - logits_40x_binary: 40X流的二分类logits
                - logits_100x_binary: 100X流的二分类logits
                - logits_final_binary: 融合后的二分类logits
                八分类：
                - logits_40x_multiclass: 40X流的八分类logits
                - logits_100x_multiclass: 100X流的八分类logits
                - logits_final_multiclass: 融合后的八分类logits
                特征：
                - features_40x: 40X特征向量（用于MMD损失）
                - features_100x: 100X特征向量（用于MMD损失）
        """
        # ========== 40X流 ==========
        # 特征提取（Swin输出特征向量）
        feat_vec_40x = self.backbone_40x(x_40x)  # (B, C_40)

        # 特征投影（统一到feature_dim，用于MMD对齐）
        v_40x = self.feature_proj_40x(feat_vec_40x)  # (B, feature_dim)
        
        # 二分类
        logits_40x_binary = self.classifier_40x_binary(v_40x)  # (B, num_classes_binary)
        
        # 八分类
        if self.use_multiclass:
            logits_40x_multiclass = self.classifier_40x_multiclass(v_40x)  # (B, num_classes_multiclass)
        else:
            logits_40x_multiclass = None
        
        # ========== 100X流 ==========
        # 特征提取（Swin输出特征向量）
        feat_vec_100x = self.backbone_100x(x_100x)  # (B, C) - Swin直接输出特征向量
        
        # 特征投影（统一到feature_dim，与40X流对齐，用于MMD损失）
        v_100x = self.feature_proj_100x(feat_vec_100x)  # (B, feature_dim)
        
        # 分类（使用投影后的特征）
        v_100x_for_classifier = self.feature_proj_100x_classifier(v_100x)  # (B, swin_feature_dim)
        
        # 二分类
        logits_100x_binary = self.classifier_100x_binary(v_100x_for_classifier)  # (B, num_classes_binary)
        
        # 八分类
        if self.use_multiclass:
            logits_100x_multiclass = self.classifier_100x_multiclass(v_100x_for_classifier)  # (B, num_classes_multiclass)
        else:
            logits_100x_multiclass = None
        
        # ========== 决策层融合（Late Fusion） ==========
        # 二分类元分类器
        prob_40x_binary = torch.softmax(logits_40x_binary, dim=1)  # (B, num_classes_binary)
        prob_100x_binary = torch.softmax(logits_100x_binary, dim=1)  # (B, num_classes_binary)
        prob_concat_binary = torch.cat([prob_40x_binary, prob_100x_binary], dim=1)  # (B, num_classes_binary*2)
        logits_final_binary = self.meta_classifier_binary(prob_concat_binary)  # (B, num_classes_binary)
        
        # 八分类元分类器
        if self.use_multiclass:
            prob_40x_multiclass = torch.softmax(logits_40x_multiclass, dim=1)  # (B, num_classes_multiclass)
            prob_100x_multiclass = torch.softmax(logits_100x_multiclass, dim=1)  # (B, num_classes_multiclass)
            prob_concat_multiclass = torch.cat([prob_40x_multiclass, prob_100x_multiclass], dim=1)  # (B, num_classes_multiclass*2)
            logits_final_multiclass = self.meta_classifier_multiclass(prob_concat_multiclass)  # (B, num_classes_multiclass)
        else:
            logits_final_multiclass = None
        
        return {
            # 二分类
            'logits_40x_binary': logits_40x_binary,
            'logits_100x_binary': logits_100x_binary,
            'logits_final_binary': logits_final_binary,
            # 八分类
            'logits_40x_multiclass': logits_40x_multiclass,
            'logits_100x_multiclass': logits_100x_multiclass,
            'logits_final_multiclass': logits_final_multiclass,
            # 特征
            'features_40x': v_40x,  # 用于MMD损失
            'features_100x': v_100x  # 用于MMD损失
        }


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("测试 DAF-MMD Net")
    print("="*60)
    
    # 创建模型（多任务学习）
    model = DAF_MMD_Net(
        num_classes_binary=2,
        num_classes_multiclass=8,
        use_multiclass=True,
        feature_dim=2048,
        swin_feature_dim=1024,
        use_pretrained=False  # 测试时不加载预训练权重
    )
    
    # 创建测试数据
    batch_size = 4
    x_40x = torch.randn(batch_size, 3, 224, 224)
    x_100x = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\n输入形状:")
    print(f"  40X: {x_40x.shape}")
    print(f"  100X: {x_100x.shape}")
    
    # 前向传播
    outputs = model(x_40x, x_100x)
    
    print(f"\n输出形状:")
    print(f"  Binary Classification:")
    print(f"    logits_40x_binary: {outputs['logits_40x_binary'].shape}")
    print(f"    logits_100x_binary: {outputs['logits_100x_binary'].shape}")
    print(f"    logits_final_binary: {outputs['logits_final_binary'].shape}")
    print(f"  8-class Classification:")
    print(f"    logits_40x_multiclass: {outputs['logits_40x_multiclass'].shape}")
    print(f"    logits_100x_multiclass: {outputs['logits_100x_multiclass'].shape}")
    print(f"    logits_final_multiclass: {outputs['logits_final_multiclass'].shape}")
    print(f"  Features:")
    print(f"    features_40x: {outputs['features_40x'].shape}")
    print(f"    features_100x: {outputs['features_100x'].shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n参数量:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    print("\n✓ DAF-MMD Net 测试通过！")

