"""
SimpleCNN 多任务学习版本 (Group 1)
支持同时进行 2分类（良性/恶性）和 8分类（具体类别）
"""
import torch
import torch.nn as nn


class SimpleCNNMultiTask(nn.Module):
    """
    简单的卷积神经网络，支持多任务学习
    """
    def __init__(self, num_classes_binary=2, num_classes_multiclass=8, 
                 dropout=0.5, use_multiclass=True):
        """
        Args:
            num_classes_binary: 二分类类别数（默认2：良性/恶性）
            num_classes_multiclass: 八分类类别数（默认8）
            dropout: Dropout比率
            use_multiclass: 是否启用八分类任务
        """
        super(SimpleCNNMultiTask, self).__init__()
        
        self.num_classes_binary = num_classes_binary
        self.num_classes_multiclass = num_classes_multiclass
        self.use_multiclass = use_multiclass
        
        # 特征提取层（共享）
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 共享的特征提取层
        self.shared_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        
        # 二分类分类头
        self.classifier_binary = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes_binary)
        )
        
        # 八分类分类头（如果启用）
        if use_multiclass:
            self.classifier_multiclass = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(256, num_classes_multiclass)
            )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.shared_fc(x)
        
        # 二分类
        logits_binary = self.classifier_binary(x)
        
        # 八分类（如果启用）
        logits_multiclass = None
        if self.use_multiclass:
            logits_multiclass = self.classifier_multiclass(x)
        
        return {
            'logits_binary': logits_binary,
            'logits_multiclass': logits_multiclass
        }


if __name__ == "__main__":
    # 测试模型
    print("测试 SimpleCNN 多任务模型:")
    model = SimpleCNNMultiTask(
        num_classes_binary=2,
        num_classes_multiclass=8,
        use_multiclass=True
    )
    
    x = torch.randn(4, 3, 224, 224)
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Binary output shape: {outputs['logits_binary'].shape}")
    print(f"Multiclass output shape: {outputs['logits_multiclass'].shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
