import numpy as np
import torch


def compute_entropy(attn: torch.Tensor) -> dict:
    """计算注意力熵"""
    attn = attn.cpu().numpy()
    seq_len = attn.shape[0]

    entropies = []
    for i in range(seq_len):
        row = attn[i].copy()
        row = row / (row.sum() + 1e-8)
        entropy = -np.sum(row * np.log(row + 1e-8))
        entropies.append(entropy)

    avg_entropy = np.mean(entropies)

    return {
        'avg_entropy': avg_entropy,
        'max_entropy': np.log(seq_len),
        'equiv_uniform': np.exp(avg_entropy),
    }


def compute_distance_distribution(attn: torch.Tensor) -> dict:
    """计算注意力距离分布"""
    attn = attn.cpu().numpy()
    seq_len = attn.shape[0]

    local, medium, long_range = 0, 0, 0
    total_weight = 0

    for i in range(seq_len):
        for j in range(seq_len):
            if i == j:
                continue
            dist = abs(i - j)
            w = attn[i, j]
            if dist <= 3:
                local += w
            elif dist <= 20:
                medium += w
            else:
                long_range += w
            total_weight += w

    return {
        'local_ratio': local / total_weight,
        'medium_ratio': medium / total_weight,
        'long_ratio': long_range / total_weight,
    }


def compute_diag_offdiag(attn: torch.Tensor) -> dict:
    """计算对角线vs非对角线注意力"""
    attn = attn.cpu().numpy()

    diag_sum = np.trace(attn)
    off_diag_sum = attn.sum() - diag_sum
    total = diag_sum + off_diag_sum

    return {
        'diag_ratio': diag_sum / total,
        'off_diag_ratio': off_diag_sum / total,
    }


def analyze_attention(attn: torch.Tensor) -> dict:
    """综合分析注意力"""
    return {
        **compute_entropy(attn),
        **compute_distance_distribution(attn),
        **compute_diag_offdiag(attn),
    }


def print_analysis(attn: torch.Tensor):
    """打印注意力分析结果"""
    stats = analyze_attention(attn)

    print(f"平均熵: {stats['avg_entropy']:.3f} (最大: {stats['max_entropy']:.3f})")
    print(f"相当于均匀分布在 ~{stats['equiv_uniform']:.1f} 个位置上")
    print(f"局部 (≤3):   {stats['local_ratio']:.1%}")
    print(f"中距离 (4-20): {stats['medium_ratio']:.1%}")
    print(f"长距离 (>20): {stats['long_ratio']:.1%}")
    print(f"对角线: {stats['diag_ratio']:.1%}")
    print(f"非对角线: {stats['off_diag_ratio']:.1%}")


attns = torch.load('attention_weight.pt')  # (8, 74, 74)

for i, attn in enumerate(attns):
    print(i)
    print_analysis(attn)
    print()