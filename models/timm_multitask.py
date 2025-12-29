"""
基于 timm 库的多任务学习模型封装
支持各种 backbone (如 resnet50, xception 等)
同时训练 2分类（良性/恶性）和 8分类（具体类别）
"""
import torch
import torch.nn as nn
import timm

class TimmMultiTaskModel(nn.Module):
    """
    使用 timm 库中的模型作为 backbone 的多任务模型
    """
    def __init__(self, model_name='resnet50', num_classes_binary=2, 
                 num_classes_multiclass=8, dropout=0.5, use_multiclass=True,
                 pretrained=True):
        """
        Args:
            model_name: timm 模型名称
            num_classes_binary: 二分类类别数
            num_classes_multiclass: 八分类类别数
            dropout: Dropout比率
            use_multiclass: 是否启用八分类任务
            pretrained: 是否使用预训练权重
        """
        super(TimmMultiTaskModel, self).__init__()
        
        self.use_multiclass = use_multiclass
        
        # 创建 backbone，移除分类头
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        # 获取特征维度
        # 有些模型返回特征维度的方式不同，通常可以通过 num_features 获得
        if hasattr(self.backbone, 'num_features'):
            self.feature_dim = self.backbone.num_features
        else:
            # 尝试通过模拟前向传播获取维度
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy_input)
                self.feature_dim = features.shape[1]
        
        # 共享的全连接层 (可选，为了与 simple_cnn 保持一致)
        self.shared_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
        )
        
        # 二分类分类头
        self.classifier_binary = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes_binary)
        )
        
        # 八分类分类头
        if use_multiclass:
            self.classifier_multiclass = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(512, num_classes_multiclass)
            )
            
    def forward(self, x):
        """
        前向传播
        """
        # 特征提取
        features = self.backbone(x)
        
        # 共享层
        x = self.shared_fc(features)
        
        # 二分类
        logits_binary = self.classifier_binary(x)
        
        # 八分类
        logits_multiclass = None
        if self.use_multiclass:
            logits_multiclass = self.classifier_multiclass(x)
            
        return {
            'logits_binary': logits_binary,
            'logits_multiclass': logits_multiclass
        }

def get_timm_multitask_model(model_name, num_classes_binary=2, 
                            num_classes_multiclass=8, dropout=0.5, 
                            use_multiclass=True, pretrained=True):
    """便捷获取模型的函数"""
    return TimmMultiTaskModel(
        model_name=model_name,
        num_classes_binary=num_classes_binary,
        num_classes_multiclass=num_classes_multiclass,
        dropout=dropout,
        use_multiclass=use_multiclass,
        pretrained=pretrained
    )

