"""
DAF-MMD Net with Xception: 将Xception融入DAF-MMD框架
使用Xception替代或补充Swin Transformer，可能带来性能提升
"""
import torch
import torch.nn as nn
import timm


class DAF_MMD_Net_Xception(nn.Module):
    """
    DAF-MMD Net模型 - Xception版本
    
    架构改进：
    1. 40X流：Xception (替代Swin Transformer) -> V_40X
    2. 100X流：Swin Transformer (保持) -> V_100X
    3. 或者双Xception流：40X和100X都使用Xception
    4. MMD对齐 + 决策层融合
    
    优势：
    - Xception在医学图像分类任务上表现优异
    - Depthwise Separable Convolution减少参数量
    - 可以更好地捕获细粒度特征
    """
    def __init__(
        self,
        num_classes_binary=2,
        num_classes_multiclass=8,
        feature_dim=2048,  # 用于MMD对齐的统一特征维度
        xception_feature_dim=2048,  # Xception的输出特征维度
        swin_feature_dim=1024,  # Swin Transformer的输出特征维度
        dropout=0.5,
        use_pretrained=True,
        use_multiclass=True,
        stream_config='xception_swin'  # 'xception_swin' 或 'xception_xception'
    ):
        """
        Args:
            stream_config: 
                - 'xception_swin': 40X用Xception, 100X用Swin
                - 'xception_xception': 40X和100X都用Xception
        """
        super(DAF_MMD_Net_Xception, self).__init__()
        
        self.num_classes_binary = num_classes_binary
        self.num_classes_multiclass = num_classes_multiclass
        self.use_multiclass = use_multiclass
        self.stream_config = stream_config
        
        # ========== 40X流：Xception ==========
        xception_40x = timm.create_model(
            'xception',
            pretrained=use_pretrained,
            num_classes=0,  # 移除分类头，返回特征
            global_pool='avg'
        )
        self.backbone_40x = xception_40x
        
        # Xception的输出特征维度
        if hasattr(self.backbone_40x, 'num_features'):
            xception_40x_out_channels = self.backbone_40x.num_features
        else:
            # 动态获取：Xception通常是2048
            with torch.no_grad():
                dummy = torch.randn(1, 3, 299, 299)  # Xception输入是299x299
                feat = self.backbone_40x(dummy)
                xception_40x_out_channels = feat.shape[1]
        
        # 特征维度投影（统一到feature_dim用于MMD对齐）
        self.feature_proj_40x = nn.Linear(xception_40x_out_channels, feature_dim)
        
        # ========== 100X流：根据配置选择 ==========
        if stream_config == 'xception_xception':
            # 100X流也用Xception
            xception_100x = timm.create_model(
                'xception',
                pretrained=use_pretrained,
                num_classes=0,
                global_pool='avg'
            )
            self.backbone_100x = xception_100x
            xception_100x_out_channels = xception_40x_out_channels  # 相同架构
            self.feature_proj_100x = nn.Linear(xception_100x_out_channels, feature_dim)
            self.feature_proj_100x_classifier = nn.Linear(feature_dim, xception_feature_dim)
            classifier_feature_dim = xception_feature_dim
        else:
            # 100X流用Swin Transformer（原始配置）
            swin = timm.create_model(
                'swin_base_patch4_window7_224',
                pretrained=use_pretrained,
                num_classes=0
            )
            self.backbone_100x = swin
            swin_out_channels = self.backbone_100x.num_features
            self.feature_proj_100x = nn.Linear(swin_out_channels, feature_dim)
            self.feature_proj_100x_classifier = nn.Linear(feature_dim, swin_feature_dim)
            classifier_feature_dim = swin_feature_dim
        
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
            nn.Linear(classifier_feature_dim, 256),
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
                nn.Linear(classifier_feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, num_classes_multiclass)
            )
        
        # ========== 元分类器（Late Fusion） ==========
        # 二分类元分类器
        meta_input_dim_binary = num_classes_binary * 2
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
            meta_input_dim_multiclass = num_classes_multiclass * 2
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
            x_40x: 40X图像 (B, 3, H, W) - 如果是Xception，需要299x299
            x_100x: 100X图像 (B, 3, H, W) - 如果是Swin，需要224x224；如果是Xception，需要299x299
        
        Returns:
            dict包含二分类、八分类logits和特征
        """
        # ========== 40X流：Xception ==========
        feat_vec_40x = self.backbone_40x(x_40x)  # (B, C_40)
        v_40x = self.feature_proj_40x(feat_vec_40x)  # (B, feature_dim)
        
        logits_40x_binary = self.classifier_40x_binary(v_40x)
        logits_40x_multiclass = None
        if self.use_multiclass:
            logits_40x_multiclass = self.classifier_40x_multiclass(v_40x)
        
        # ========== 100X流 ==========
        feat_vec_100x = self.backbone_100x(x_100x)
        v_100x = self.feature_proj_100x(feat_vec_100x)  # (B, feature_dim)
        
        if self.stream_config == 'xception_xception':
            v_100x_for_classifier = self.feature_proj_100x_classifier(v_100x)
        else:
            v_100x_for_classifier = self.feature_proj_100x_classifier(v_100x)
        
        logits_100x_binary = self.classifier_100x_binary(v_100x_for_classifier)
        logits_100x_multiclass = None
        if self.use_multiclass:
            logits_100x_multiclass = self.classifier_100x_multiclass(v_100x_for_classifier)
        
        # ========== 决策层融合 ==========
        prob_40x_binary = torch.softmax(logits_40x_binary, dim=1)
        prob_100x_binary = torch.softmax(logits_100x_binary, dim=1)
        prob_concat_binary = torch.cat([prob_40x_binary, prob_100x_binary], dim=1)
        logits_final_binary = self.meta_classifier_binary(prob_concat_binary)
        
        logits_final_multiclass = None
        if self.use_multiclass:
            prob_40x_multiclass = torch.softmax(logits_40x_multiclass, dim=1)
            prob_100x_multiclass = torch.softmax(logits_100x_multiclass, dim=1)
            prob_concat_multiclass = torch.cat([prob_40x_multiclass, prob_100x_multiclass], dim=1)
            logits_final_multiclass = self.meta_classifier_multiclass(prob_concat_multiclass)
        
        return {
            'logits_40x_binary': logits_40x_binary,
            'logits_100x_binary': logits_100x_binary,
            'logits_final_binary': logits_final_binary,
            'logits_40x_multiclass': logits_40x_multiclass,
            'logits_100x_multiclass': logits_100x_multiclass,
            'logits_final_multiclass': logits_final_multiclass,
            'features_40x': v_40x,
            'features_100x': v_100x
        }


def get_daf_mmd_xception_model(
    stream_config='xception_swin',
    num_classes_binary=2,
    num_classes_multiclass=8,
    use_multiclass=True,
    use_pretrained=True
):
    """便捷获取模型的函数"""
    return DAF_MMD_Net_Xception(
        num_classes_binary=num_classes_binary,
        num_classes_multiclass=num_classes_multiclass,
        use_multiclass=use_multiclass,
        use_pretrained=use_pretrained,
        stream_config=stream_config
    )


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("测试 DAF-MMD Net with Xception")
    print("="*60)
    
    # 测试配置1：40X用Xception，100X用Swin
    print("\n配置1: 40X-Xception + 100X-Swin")
    model1 = get_daf_mmd_xception_model(stream_config='xception_swin')
    
    x_40x = torch.randn(2, 3, 299, 299)  # Xception需要299x299
    x_100x = torch.randn(2, 3, 224, 224)  # Swin需要224x224
    
    outputs1 = model1(x_40x, x_100x)
    print(f"  Binary logits shape: {outputs1['logits_final_binary'].shape}")
    print(f"  Multiclass logits shape: {outputs1['logits_final_multiclass'].shape}")
    
    # 测试配置2：双Xception流
    print("\n配置2: 40X-Xception + 100X-Xception")
    model2 = get_daf_mmd_xception_model(stream_config='xception_xception')
    
    x_40x = torch.randn(2, 3, 299, 299)
    x_100x = torch.randn(2, 3, 299, 299)  # Xception也需要299x299
    
    outputs2 = model2(x_40x, x_100x)
    print(f"  Binary logits shape: {outputs2['logits_final_binary'].shape}")
    print(f"  Multiclass logits shape: {outputs2['logits_final_multiclass'].shape}")
    
    print("\n✓ DAF-MMD Net with Xception 测试通过！")




