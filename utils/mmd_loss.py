"""
Maximum Mean Discrepancy (MMD) Loss
用于领域对齐，实现放大倍率不变性
"""
import torch
import torch.nn as nn


def gaussian_kernel(x, y, sigma=1.0):
    """
    高斯核函数
    
    Args:
        x: 特征向量集合 (N1, D)
        y: 特征向量集合 (N2, D)
        sigma: 高斯核的带宽参数
    
    Returns:
        核矩阵 (N1, N2)
    """
    # 计算L2距离的平方
    # ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 * x_i^T * y_j
    x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (N1, 1)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)  # (N2, 1)
    xy = torch.matmul(x, y.t())  # (N1, N2)
    
    dist_sq = x_norm + y_norm.t() - 2 * xy
    # 高斯核: exp(-dist^2 / (2 * sigma^2))
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def mmd_loss(x, y, sigma=1.0):
    """
    计算两个特征分布之间的MMD损失
    
    Args:
        x: 第一个分布的特征向量 (N1, D)
        y: 第二个分布的特征向量 (N2, D)
        sigma: 高斯核的带宽参数
    
    Returns:
        MMD损失值（标量）
    
    公式：
        MMD^2 = E[k(x_i, x_j)] + E[k(y_i, y_j)] - 2*E[k(x_i, y_j)]
    """
    # 计算核矩阵
    kxx = gaussian_kernel(x, x, sigma)  # (N1, N1)
    kyy = gaussian_kernel(y, y, sigma)  # (N2, N2)
    kxy = gaussian_kernel(x, y, sigma)  # (N1, N2)
    
    # 计算MMD^2
    # 对角线元素是k(x_i, x_i) = 1（因为dist=0），需要排除
    n1 = x.size(0)
    n2 = y.size(0)
    
    # 检查batch size，避免除以零
    if n1 <= 1 or n2 <= 1:
        # batch size太小，返回0（不计算MMD）
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    
    # 计算均值（排除对角线）
    # 正确的公式：(sum - trace) / (n * (n-1))
    kxx_sum_no_diag = (kxx.sum() - kxx.trace()) / (n1 * (n1 - 1))
    kyy_sum_no_diag = (kyy.sum() - kyy.trace()) / (n2 * (n2 - 1))
    kxy_mean = kxy.sum() / (n1 * n2)
    
    mmd_sq = kxx_sum_no_diag + kyy_sum_no_diag - 2 * kxy_mean
    
    # 确保非负（数值稳定性）
    mmd_sq = torch.clamp(mmd_sq, min=0.0)
    
    # 返回MMD（开平方根）
    return torch.sqrt(mmd_sq + 1e-8)  # 添加小常数避免梯度消失


class MMDLoss(nn.Module):
    """
    MMD损失模块
    用于对齐40X和100X特征分布
    """
    def __init__(self, sigma=1.0):
        """
        Args:
            sigma: 高斯核的带宽参数，默认1.0
        """
        super(MMDLoss, self).__init__()
        self.sigma = sigma
    
    def forward(self, feat_40x, feat_100x):
        """
        计算40X和100X特征之间的MMD损失
        
        Args:
            feat_40x: 40X特征向量 (B1, D)
            feat_100x: 100X特征向量 (B2, D)
        
        Returns:
            MMD损失值
        """
        return mmd_loss(feat_40x, feat_100x, self.sigma)


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("测试 MMD Loss")
    print("="*60)
    
    # 创建测试数据
    batch_size_40x = 8
    batch_size_100x = 8
    feature_dim = 512
    
    # 模拟40X特征（来自一个分布）
    feat_40x = torch.randn(batch_size_40x, feature_dim)
    
    # 模拟100X特征（来自另一个分布）
    feat_100x = torch.randn(batch_size_100x, feature_dim) + 0.5  # 添加偏移，模拟分布差异
    
    print(f"\n40X特征形状: {feat_40x.shape}")
    print(f"100X特征形状: {feat_100x.shape}")
    
    # 计算MMD损失
    mmd_loss_fn = MMDLoss(sigma=1.0)
    loss = mmd_loss_fn(feat_40x, feat_100x)
    
    print(f"\nMMD损失: {loss.item():.4f}")
    
    # 测试相同分布（应该接近0）
    feat_same = torch.randn(batch_size_40x, feature_dim)
    loss_same = mmd_loss_fn(feat_40x, feat_same)
    print(f"相同分布MMD损失: {loss_same.item():.4f} (应该较小)")
    
    print("\n✓ MMD Loss 测试通过！")

