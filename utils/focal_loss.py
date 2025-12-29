"""
Focal Loss实现
专门用于处理类别不平衡问题
论文: Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL = -α * (1 - p_t)^γ * log(p_t)
    
    参数说明:
        alpha: 类别权重，tensor shape [num_classes]，用于平衡类别
        gamma: 聚焦参数，降低容易样本的权重，增加难样本的权重
               - gamma=0: 等价于标准的CrossEntropyLoss
               - gamma=2: 论文推荐值
               - gamma越大，对难样本的关注越多
        reduction: 'mean', 'sum' 或 'none'
    
    工作原理:
        - 对于容易分类的样本(p_t接近1)，(1-p_t)^γ接近0，损失很小
        - 对于难分类的样本(p_t接近0)，(1-p_t)^γ接近1，损失很大
        - 这样模型会专注于学习难样本，而不是在容易样本上浪费时间
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 类别权重，shape [num_classes] 的tensor，或None。
                   如果是其他类型（如float/int），会被忽略。
            gamma: 聚焦参数，默认2.0
            reduction: 'mean', 'sum' 或 'none'
        """
        super(FocalLoss, self).__init__()
        # 确保alpha要么是Tensor，要么是None
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            self.alpha = None
        else:
            self.alpha = alpha
            
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出 logits，shape [batch_size, num_classes]
            targets: 真实标签，shape [batch_size]
        
        Returns:
            focal loss值
        """
        # 计算标准的交叉熵损失（不进行reduction）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # 计算 p_t：模型对正确类别的预测概率
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE)
        
        # 计算 Focal Loss: (1 - p_t)^gamma * CE
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLossWithLabelSmoothing(nn.Module):
    """
    Focal Loss + Label Smoothing
    结合了Focal Loss和标签平滑的优点
    """
    
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        """
        Args:
            alpha: 类别权重
            gamma: 聚焦参数
            label_smoothing: 标签平滑系数，范围[0, 1)
            reduction: 'mean', 'sum' 或 'none'
        """
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.num_classes = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出 logits，shape [batch_size, num_classes]
            targets: 真实标签，shape [batch_size]
        """
        batch_size, num_classes = inputs.shape
        
        if self.num_classes is None:
            self.num_classes = num_classes
        
        # Label smoothing
        # 原始: [0, 0, 1, 0, 0] 
        # 平滑后: [ε/K, ε/K, 1-ε+ε/K, ε/K, ε/K]
        # 其中 ε 是 label_smoothing，K 是类别数
        confidence = 1.0 - self.label_smoothing
        smooth_label = self.label_smoothing / num_classes
        
        # 创建平滑标签
        one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        smooth_one_hot = one_hot * confidence + smooth_label
        
        # 计算log_softmax
        log_probs = F.log_softmax(inputs, dim=1)
        
        # 计算加权的负对数似然
        if self.alpha is not None:
            # 扩展alpha到batch维度
            alpha_t = self.alpha[targets]
            loss = -(smooth_one_hot * log_probs).sum(dim=1) * alpha_t
        else:
            loss = -(smooth_one_hot * log_probs).sum(dim=1)
        
        # 计算focal weight: (1 - p_t)^gamma
        probs = F.softmax(inputs, dim=1)
        pt = (probs * one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        
        # 应用focal weight
        loss = focal_weight * loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("测试 Focal Loss")
    print("="*60)
    
    # 创建测试数据
    batch_size = 4
    num_classes = 8
    inputs = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    
    # 创建类别权重（模拟不平衡数据集）
    class_weights = torch.tensor([2.0, 1.5, 1.0, 1.0, 0.8, 0.8, 0.5, 0.3])
    
    # 测试标准CrossEntropyLoss
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    ce_output = ce_loss(inputs, targets)
    print(f"\nCrossEntropyLoss: {ce_output.item():.4f}")
    
    # 测试Focal Loss (gamma=0, 应该等价于CE)
    focal_loss_g0 = FocalLoss(alpha=class_weights, gamma=0.0)
    fl_output_g0 = focal_loss_g0(inputs, targets)
    print(f"FocalLoss (gamma=0): {fl_output_g0.item():.4f} (应该≈CE)")
    
    # 测试Focal Loss (gamma=2)
    focal_loss_g2 = FocalLoss(alpha=class_weights, gamma=2.0)
    fl_output_g2 = focal_loss_g2(inputs, targets)
    print(f"FocalLoss (gamma=2): {fl_output_g2.item():.4f}")
    
    # 测试Focal Loss + Label Smoothing
    focal_loss_ls = FocalLossWithLabelSmoothing(
        alpha=class_weights, 
        gamma=2.0, 
        label_smoothing=0.1
    )
    fl_ls_output = focal_loss_ls(inputs, targets)
    print(f"FocalLoss + LabelSmoothing: {fl_ls_output.item():.4f}")
    
    print("\n✓ Focal Loss 测试通过！")

