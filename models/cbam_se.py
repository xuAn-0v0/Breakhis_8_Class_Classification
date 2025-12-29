"""
CBAM-SE Attention Module
结合了通道注意力（Channel Attention）和空间注意力（Spatial Attention）
以及Squeeze-and-Excitation机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM_SE(nn.Module):
    """
    CBAM-SE融合注意力模块
    结合了CBAM（通道+空间注意力）和SE（Squeeze-and-Excitation）机制
    
    流程：
    1. 输入特征图 F
    2. 通道注意力 -> F' = F * ChannelAttention(F)
    3. 空间注意力 -> F'' = F' * SpatialAttention(F')
    4. SE机制增强 -> 最终输出
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        """
        Args:
            in_channels: 输入通道数
            reduction: 通道压缩比例（用于SE和ChannelAttention）
            kernel_size: 空间注意力卷积核大小
        """
        super(CBAM_SE, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
        # SE模块：全局平均池化 + FC层
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        
        Returns:
            增强后的特征图 (B, C, H, W)
        """
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 空间注意力
        sa = self.spatial_attention(x)
        x = x * sa
        
        # SE机制增强
        se = self.se(x)
        x = x * se
        
        return x


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("测试 CBAM-SE 模块")
    print("="*60)
    
    # 创建测试数据
    batch_size = 4
    channels = 512
    height, width = 7, 7
    
    x = torch.randn(batch_size, channels, height, width)
    print(f"\n输入形状: {x.shape}")
    
    # 创建CBAM-SE模块
    cbam_se = CBAM_SE(in_channels=channels, reduction=16)
    
    # 前向传播
    output = cbam_se(x)
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in cbam_se.parameters()):,}")
    
    print("\n✓ CBAM-SE 模块测试通过！")

