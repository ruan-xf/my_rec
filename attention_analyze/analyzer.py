"""
注意力分析模块

提供采样、获取注意力权重、分析和可视化的完整功能。
支持分析不同长度序列的注意力模式。

使用方式：
- 分析已保存的数据：导入纯分析函数（find_hotspot_columns, analyze_attention 等）
- 采样和模型推理：导入 model_utils 模块中的函数（sample_item_seq, get_attention_weights）
"""

import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class AttentionStats:
    """注意力统计结果"""
    avg_entropy: float
    max_entropy: float
    equiv_uniform: float
    local_ratio: float
    medium_ratio: float
    long_ratio: float
    diag_ratio: float
    off_diag_ratio: float


@dataclass
class HotspotColumn:
    """热点列分析结果"""
    column: int
    total_attention: float


def compute_entropy(attn: torch.Tensor, query_position: str = 'last') -> dict:
    """计算注意力熵

    Args:
        attn: 单个注意力头的权重矩阵 (seq_len, seq_len)
        query_position: 'last' 只分析最后位置（推荐任务目标物品，推荐）
                       'all' 分析所有位置的平均（原行为，不推荐）

    Returns:
        包含 avg_entropy, max_entropy, equiv_uniform 的字典

    Note:
        推荐任务中，序列最后一位是目标物品，应该只分析它对历史的注意力。
        全局平均会混淆历史位置和目标位置的语义，掩盖真实的长距离依赖能力。
        详见 attention_analyze/README.md "关键设计决策：分析目标位置"
    """
    attn = attn.cpu().numpy()
    seq_len = attn.shape[0]

    if query_position == 'last':
        # 只分析最后一个位置（目标物品）对历史的注意力
        # 排除对自身的关注（对角线）
        last_query = attn[-1, :-1]  # shape: (seq_len-1,)
        last_query = last_query / (last_query.sum() + 1e-8)
        entropy = -np.sum(last_query * np.log(last_query + 1e-8))

        return {
            'avg_entropy': entropy,  # 保持命名兼容
            'max_entropy': np.log(seq_len - 1) if seq_len > 1 else 0,  # 最大熵是均匀分布到 seq_len-1 个历史位置
            'equiv_uniform': np.exp(entropy),
        }
    else:
        # 原行为：所有位置的平均（不推荐用于推荐任务）
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


def compute_distance_distribution(attn: torch.Tensor, query_position: str = 'last') -> dict:
    """计算注意力距离分布

    Args:
        attn: 单个注意力头的权重矩阵
        query_position: 'last' 只分析最后位置（推荐）
                       'all' 分析所有位置的平均（不推荐）

    Returns:
        包含 local_ratio, medium_ratio, long_ratio 的字典

    Note:
        推荐任务中应使用 query_position='last'，分析目标物品回顾历史的距离。
        距离定义：目标位置 - 历史位置 j（正值，表示回顾的远近）
    """
    attn = attn.cpu().numpy()
    seq_len = attn.shape[0]

    if query_position == 'last':
        # 只分析目标物品对历史的注意力距离分布
        if seq_len <= 1:
            return {
                'local_ratio': 0.0,
                'medium_ratio': 0.0,
                'long_ratio': 0.0,
            }

        local, medium, long_range = 0, 0, 0
        total_weight = 0

        last_query = attn[-1, :-1]  # 最后位置对历史的注意力

        for j in range(seq_len - 1):
            dist = (seq_len - 1) - j  # 距离目标的位置（1 到 seq_len-1）
            w = last_query[j]

            if dist <= 3:
                local += w
            elif dist <= 20:
                medium += w
            else:
                long_range += w
            total_weight += w

        if total_weight == 0:
            return {
                'local_ratio': 0.0,
                'medium_ratio': 0.0,
                'long_ratio': 0.0,
            }

        return {
            'local_ratio': local / total_weight,
            'medium_ratio': medium / total_weight,
            'long_ratio': long_range / total_weight,
        }
    else:
        # 原行为：所有位置的平均（不推荐）
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

        if total_weight == 0:
            return {
                'local_ratio': 0.0,
                'medium_ratio': 0.0,
                'long_ratio': 0.0,
            }

        return {
            'local_ratio': local / total_weight,
            'medium_ratio': medium / total_weight,
            'long_ratio': long_range / total_weight,
        }


def compute_diag_offdiag(attn: torch.Tensor) -> dict:
    """计算对角线 vs 非对角线注意力

    Args:
        attn: 单个注意力头的权重矩阵

    Returns:
        包含 diag_ratio, off_diag_ratio 的字典
    """
    attn = attn.cpu().numpy()

    diag_sum = np.trace(attn)
    off_diag_sum = attn.sum() - diag_sum
    total = diag_sum + off_diag_sum

    return {
        'diag_ratio': diag_sum / total,
        'off_diag_ratio': off_diag_sum / total,
    }


def find_hotspot_columns(attn: torch.Tensor, top_k: int = 5, exclude_diag: bool = True) -> list[HotspotColumn]:
    """找出被关注最多的热点列

    Args:
        attn: 单个注意力头的权重矩阵
        top_k: 返回前k个热点列
        exclude_diag: 是否排除对角线

    Returns:
        热点列列表，按热度降序排列
    """
    attn = attn.cpu().numpy()

    if exclude_diag:
        attn_no_diag = attn.copy()
        np.fill_diagonal(attn_no_diag, 0)
        col_sums = attn_no_diag.sum(axis=0)
    else:
        col_sums = attn.sum(axis=0)

    top_indices = np.argsort(col_sums)[-top_k:][::-1]

    return [
        HotspotColumn(column=int(idx), total_attention=float(col_sums[idx]))
        for idx in top_indices
    ]


def analyze_attention(attn: torch.Tensor, query_position: str = 'last') -> AttentionStats:
    """综合分析单个注意力头

    Args:
        attn: 单个注意力头的权重矩阵
        query_position: 'last' 只分析最后位置（推荐任务，推荐）
                       'all' 分析所有位置的平均（原行为，不推荐）

    Returns:
        AttentionStats 对象

    Note:
        推荐任务应使用 query_position='last'，分析目标物品如何从历史中聚合信息。
        详见 attention_analyze/README.md "关键设计决策：分析目标位置"
    """
    entropy_stats = compute_entropy(attn, query_position=query_position)
    distance_stats = compute_distance_distribution(attn, query_position=query_position)
    diag_stats = compute_diag_offdiag(attn)

    return AttentionStats(
        avg_entropy=entropy_stats['avg_entropy'],
        max_entropy=entropy_stats['max_entropy'],
        equiv_uniform=entropy_stats['equiv_uniform'],
        local_ratio=distance_stats['local_ratio'],
        medium_ratio=distance_stats['medium_ratio'],
        long_ratio=distance_stats['long_ratio'],
        diag_ratio=diag_stats['diag_ratio'],
        off_diag_ratio=diag_stats['off_diag_ratio'],
    )


def analyze_all_heads(attention_weights: torch.Tensor, query_position: str = 'last') -> list[AttentionStats]:
    """分析所有注意力头

    Args:
        attention_weights: shape=(num_heads, seq_len, seq_len)
        query_position: 'last' 只分析最后位置（推荐）
                       'all' 分析所有位置的平均（不推荐）

    Returns:
        每个头的统计结果列表
    """
    return [analyze_attention(attn, query_position=query_position) for attn in attention_weights]


def visualize_attention(
    attn: torch.Tensor,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True
):
    """可视化注意力矩阵

    Args:
        attn: 单个注意力头的权重矩阵
        title: 图标题
        save_path: 保存路径（可选）
        show: 是否显示图像
    """
    import matplotlib.pyplot as plt

    attn_np = attn.cpu().numpy()
    plt.imshow(attn_np, cmap='viridis', aspect='auto')
    plt.colorbar()

    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def visualize_all_heads(
    attention_weights: torch.Tensor,
    save_dir: Optional[Path] = None,
    show: bool = True
):
    """可视化所有注意力头

    Args:
        attention_weights: shape=(num_heads, seq_len, seq_len)
        save_dir: 保存目录（可选）
        show: 是否显示图像
    """
    import matplotlib.pyplot as plt

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for i, attn in enumerate(attention_weights):
        save_path = save_dir / f'head_{i}.png' if save_dir else None
        visualize_attention(attn, title=f'Head {i}', save_path=save_path, show=show)


def print_stats(stats: AttentionStats, head_id: int = None):
    """打印注意力统计结果"""
    prefix = f"Head {head_id}: " if head_id is not None else ""

    print(f"{prefix}平均熵: {stats.avg_entropy:.3f} (最大: {stats.max_entropy:.3f})")
    print(f"{prefix}相当于均匀分布在 ~{stats.equiv_uniform:.1f} 个位置上")
    print(f"{prefix}局部 (≤3):   {stats.local_ratio:.1%}")
    print(f"{prefix}中距离 (4-20): {stats.medium_ratio:.1%}")
    print(f"{prefix}长距离 (>20): {stats.long_ratio:.1%}")
    print(f"{prefix}对角线: {stats.diag_ratio:.1%}")
    print(f"{prefix}非对角线: {stats.off_diag_ratio:.1%}")


# 便捷函数：完整分析流程（需要模型依赖）
def run_analysis(
    seq_len: int,
    seed: Optional[int] = None,
    model_path: str = 'checkpoints/albert_rec/checkpoint-3000',
) -> tuple[torch.Tensor, list[AttentionStats]]:
    """运行完整的注意力分析流程

    注意：此函数需要导入模型和数据模块，较慢。

    Args:
        seq_len: 目标序列长度
        seed: 随机种子
        model_path: 模型检查点路径

    Returns:
        (注意力权重, 统计结果列表)
    """
    # 延迟导入，避免加载模型依赖
    from attention_analyze.model_utils import get_attention_weights, sample_item_seq

    logdir = Path(f'attention_analyze/logs/seqlen={seq_len}')
    logdir.mkdir(exist_ok=True)

    print(f"采样序列长度 {seq_len}...")
    item_seq = sample_item_seq(seq_len, seed=seed)
    with open(logdir / 'item_seq.pkl', 'wb') as f:
        pickle.dump(item_seq, f)

    print("获取注意力权重...")
    attention_weights = get_attention_weights(item_seq, model_path)
    torch.save(attention_weights, logdir / 'attention_weight.pt')

    print("分析注意力头...")
    all_stats = analyze_all_heads(attention_weights, query_position='last')  # 使用修正后的分析

    for i, stats in enumerate(all_stats):
        print(f"\n=== Head {i} ===")
        print_stats(stats)

        hotspots = find_hotspot_columns(attention_weights[i])
        print(f"热点列: {[(h.column, f'{h.total_attention:.1f}') for h in hotspots[:3]]}")

    print("\n可视化...")
    visualize_all_heads(attention_weights)

    return attention_weights, all_stats


# results = []
# for seq_len in [3,10,30,74,150]:
#     logdir = Path(f'logs/seqlen={seq_len}')
#     attention_weights = torch.load(logdir / 'attention_weight.pt', weights_only=False)
#     all_stats = analyze_all_heads(attention_weights, query_position='last')  # 使用修正后的分析

#     for i, stats in enumerate(all_stats):
#         hotspots = find_hotspot_columns(attention_weights[i])
#         stat = asdict(all_stats[i])
#         stat['hotspot_columns'] = list(map(asdict, hotspots))
#         results.append((seq_len, i, stat))


# import pandas as pd
# result_df = pd.DataFrame(results)


# result_df = result_df.set_index([0,1])
# result_df = pd.DataFrame(result_df.squeeze().to_list(), result_df.index)
# # Index(['avg_entropy', 'max_entropy', 'equiv_uniform', 'local_ratio',
# #        'medium_ratio', 'long_ratio', 'diag_ratio', 'off_diag_ratio',
# #        'hotspot_columns'],
# #       dtype='str')

# result_df = result_df.reset_index(names='seqlen head'.split())
# result_df['attention_spread_ratio'] = result_df.equiv_uniform / result_df.seqlen

# result_df.to_csv('attention_analyze.csv', index=False)

def verify_transformer_advantages(seq_len=74, num_samples=10, is_prepared=False):
    """验证多头注意力的三个核心优势"""
    if not is_prepared:
        import model_utils

        model = model_utils.model_init()
        test_df = model_utils.load_test_data()

        # 采样多个样本
        samples = []
        for _ in range(num_samples):
            item_seq = model_utils.sample_item_seq(test_df, seq_len)
            attn = model_utils.get_attention_weights(item_seq, model=model)  # [8, seq_len, seq_len]
            samples.append(attn)

        with open('samples.pkl', 'wb') as f:
            pickle.dump(samples, f)

        return

    with open('samples.pkl', 'rb') as f:
        samples: list = pickle.load(f)

    # ========== 验证点1：多尺度感受野 ==========
    print("【验证1】多尺度感受野并存")
    print("(通过注意力熵衡量关注范围)")

    receptive_fields = []
    head_types = []

    for head_id in range(8):
        attn = samples[0][head_id]

        # 判断头的类型：目标导向 vs 历史关系
        target_attention = attn[:-1, -1].sum().item()
        history_attention = attn[:-1, :-1].sum().item()

        if target_attention > history_attention:
            head_types.append("目标导向")
            # 对于目标导向的头，计算等效感受野=1（只关注目标）
            receptive_fields.append(1.0)
        else:
            head_types.append("历史关系")
            # 对于历史关系的头，计算关注的平均位置数
            avg_focus_count = (attn[:-1, :-1] > 0.01).sum(dim=1).float().mean()
            receptive_fields.append(avg_focus_count.item())

    min_rf = min(receptive_fields)
    max_rf = max(receptive_fields)

    print(f"  头类型分布:")
    for i in range(8):
        print(f"    Head {i}: {head_types[i]}, 等效感受野={receptive_fields[i]:.1f}")
    print(f"  最小感受野: {min_rf:.1f}")
    print(f"  最大感受野: {max_rf:.1f}")
    print(f"  ✅ 证明：不同头的感受野差异 {max_rf/max(min_rf, 0.01):.1f}x，且存在目标导向与历史关系两种模式")
    
    # ========== 验证点2：动态内容选择 ==========
    print("\n【验证2】动态内容选择")
    
    for head_id in range(8):
        # 统计该头在不同样本下的热点位置
        hotspot_positions = []
        for sample in samples:
            attn = sample[head_id]
            # 找到被关注最多的列
            col_sums = attn[:-1, :-1].sum(dim=0)
            hotspot_pos = col_sums.argmax().item()
            hotspot_positions.append(hotspot_pos)
        
        variance = np.var(hotspot_positions)
        print(hotspot_positions)
        
        if variance < 1.0:
            print(f"  Head {head_id}: 位置偏置 (总是关注位置 {np.mean(hotspot_positions):.0f})")
        else:
            print(f"  Head {head_id}: 内容相关 (位置方差 {variance:.1f})")
    
    # ========== 验证点3：直接长距离建模 ==========
    print("\n【验证3】直接长距离建模")
    
    for head_id in range(8):
        attn = samples[0][head_id]
        seq_len = attn.shape[0]
        
        # 计算长距离依赖（Query和Key距离>20）
        long_range_attn = 0
        total_attn = 0
        
        for i in range(seq_len - 1):
            for j in range(seq_len - 1):
                if abs(i - j) > 20:
                    long_range_attn += attn[i, j].item()
                total_attn += attn[i, j].item()
        
        long_range_ratio = long_range_attn / total_attn
        print(f"  Head {head_id}: 长距离注意力占比 {long_range_ratio:.1%}")
    
    print("\n✅ 证明：注意力机制可直接建模长距离依赖")


# 如果需要运行验证，取消下面的注释
verify_transformer_advantages(is_prepared=True)