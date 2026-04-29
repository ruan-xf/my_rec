"""
序列长度性能分析 - 概率分布视角

基于 CDF/PDF 的思想，分析模型性能随序列长度的连续变化
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

import os
os.chdir('..')

# import utils


def load_test_data():
    """加载测试数据"""
    df = pl.read_parquet('data/processed/hf_saved/test.parquet')
    df_meta = df.select(
        pl.col('item_seq').list.len().alias('seq_len')
    )
    return df_meta


def load_model_output(model: str):
    """加载模型预测结果"""
    output = torch.load(f'final_results/{model}_output.pt', weights_only=False)
    return output


def compute_cumulative_performance(seqlens, errors, l_max):
    """
    计算累积性能函数 Φ(l_max) = E[error | L ≤ l_max]

    Args:
        seqlens: 序列长度数组
        errors: 误差数组 (abs error 或 squared error)
        l_max: 最大长度阈值

    Returns:
        累积性能值，如果无样本返回 None
    """
    mask = seqlens <= l_max
    if not mask.any():
        return None
    return errors[mask].mean()


def compute_performance_at_length(seqlens, errors, l):
    """
    计算特定长度的性能 M(l) = E[error | L = l]

    Args:
        seqlens: 序列长度数组
        errors: 误差数组
        l: 序列长度

    Returns:
        性能值，如果无样本返回 None
    """
    mask = seqlens == l
    if not mask.any():
        return None
    return errors[mask].mean()


def compute_performance_density(cumulative_perf_func, l, delta=1.0):
    """
    通过数值微分计算性能密度

    dΦ/dl ≈ [Φ(l + δ/2) - Φ(l - δ/2)] / δ

    Args:
        cumulative_perf_func: 累积性能函数 (接受 l_max 参数)
        l: 序列长度
        delta: 微分步长

    Returns:
        性能密度（导数）
    """
    phi_plus = cumulative_perf_func(l + delta/2)
    phi_minus = cumulative_perf_func(l - delta/2)

    if phi_plus is None or phi_minus is None:
        return None

    return (phi_plus - phi_minus) / delta


def analyze_model(model: str, df_meta: pl.DataFrame):
    """分析单个模型"""
    print(f"\n{'='*60}")
    print(f"分析模型: {model.upper()}")
    print(f"{'='*60}")

    # 加载预测结果
    if 'pred' in df_meta.columns:
        # 如果 df_meta 已包含预测结果，直接使用
        predictions = df_meta['pred'].to_numpy()
    else:
        # 否则从文件加载
        predictions = load_model_output(model)
    labels = np.ones(len(predictions))
    seqlens = df_meta['seq_len'].to_numpy()

    # 计算误差
    abs_errors = np.abs(predictions - labels)
    squared_errors = (predictions - labels) ** 2

    # 获取所有唯一的序列长度
    unique_lens = np.unique(seqlens)
    print(f"序列长度范围: {unique_lens.min()} - {unique_lens.max()}")
    print(f"唯一长度数量: {len(unique_lens)}")

    # 构建累积性能函数（闭包）
    def cumulative_perf_mae(l_max):
        return compute_cumulative_performance(seqlens, abs_errors, l_max)

    def cumulative_perf_mse(l_max):
        return compute_cumulative_performance(seqlens, squared_errors, l_max)

    # 分析每个长度点
    results = []

    # 累积变量
    cumulative_count = 0
    cumulative_prob = 0.0
    cumulative_mae_sum = 0.0  # 累积总绝对误差
    cumulative_mse_sum = 0.0  # 累积总平方误差

    for l in unique_lens:
        # 该长度的样本数
        count = (seqlens == l).sum()
        prob = count / len(seqlens)

        # 性能 M(l)
        mae_l = compute_performance_at_length(seqlens, abs_errors, l)
        mse_l = compute_performance_at_length(seqlens, squared_errors, l)

        # 累积值
        cumulative_count += count
        cumulative_prob += prob
        cumulative_mae_sum += mae_l * count  # 该长度的总绝对误差
        cumulative_mse_sum += mse_l * count  # 该长度的总平方误差

        # 累积性能 Φ(l)
        cum_mae = cumulative_perf_mae(l)
        cum_mse = cumulative_perf_mse(l)

        # 性能密度 dΦ/dl (数值微分)
        density_mae = compute_performance_density(cumulative_perf_mae, l)
        density_mse = compute_performance_density(cumulative_perf_mse, l)

        results.append({
            'seq_len': l,
            'count': count,
            'prob': prob,
            'mae': mae_l,
            'mse': mse_l,
            'cumulative_count': cumulative_count,
            'cumulative_prob': cumulative_prob,
            'cumulative_mae_sum': cumulative_mae_sum,
            'cumulative_mse_sum': cumulative_mse_sum,
            'cumulative_mae': cum_mae,
            'cumulative_mse': cum_mse,
            'density_mae': density_mae,
            'density_mse': density_mse
        })

    df_results = pd.DataFrame(results)

    return df_results


def compare_models(save_dir: Path):
    """模型对比分析"""
    models = ['albert', 'cnn', 'mlp']

    comparison_data = []
    for model in models:
        df = pd.read_csv(save_dir / f'{model}_analysis.csv')

        comparison_data.append({
            'model': model,
            'overall_mae': df['mae'].mean(),
            'overall_mse': df['mse'].mean(),
            'min_mae': df['mae'].min(),
            'max_mae': df['mae'].max(),
            'mae_std': df['mae'].std(),
            'best_len': df.loc[df['mae'].idxmin(), 'seq_len'],
            'worst_len': df.loc[df['mae'].idxmax(), 'seq_len']
        })

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv(save_dir / 'model_comparison.csv', index=False)

    print(f"\n{'='*60}")
    print("模型对比")
    print(f"{'='*60}")
    print(df_comparison.to_string(index=False))

    return df_comparison


def main():
    """主函数"""
    save_dir = Path('final_results/序列长度性能分析')
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    df_meta = load_test_data()
    print(f"总样本数: {len(df_meta)}")

    # 分析各模型
    models = ['albert', 'cnn', 'mlp']
    for model in models:
        analyze_model(model, df_meta, save_dir)

    # 模型对比
    compare_models(save_dir)

    print(f"\n{'='*60}")
    print(f"结果已保存到: {save_dir}")
    print(f"{'='*60}")


def compute_interval_integral_from_cumulative(df_analysis, interval):
    """
    利用累积总误差计算区间积分 E[error | l_min < L ≤ l_max]

    新公式（更简洁）：
    E[error | l_min < L ≤ l_max] = [S(l_max) - S(l_min)] / C(interval)

    其中：
    - S(l) = 累积总误差 = Σ|error| for L ≤ l
    - C(interval) = 区间样本数 = count(l_min < L ≤ l_max)

    Args:
        df_analysis: analyze_model 返回的 DataFrame，包含:
                     - seq_len: 序列长度
                     - cumulative_mae_sum: 累积总绝对误差 S(l)
                     - cumulative_mse_sum: 累积总平方误差
                     - cumulative_count: 累积样本数 C(l)
                     - cumulative_prob: 累积概率 P(L ≤ l)
        interval: 区间元组 (l_min, l_max)，表示 (l_min, l_max]，左开右闭

    Returns:
        dict: {
            'mae': 区间内的平均绝对误差,
            'mse': 区间内的平均平方误差,
            'count': 区间内样本数,
            'prob': 区间内样本占比
        }
    """
    l_min, l_max = interval

    # 查找区间边界对应的累积值
    # 找到 <= l_max 的最大长度
    df_max = df_analysis[df_analysis['seq_len'] <= l_max]
    if len(df_max) == 0:
        return {'mae': None, 'mse': None, 'count': 0, 'prob': 0.0}

    row_max = df_max.iloc[-1]  # 最后一行是 <= l_max 的最大长度

    # 找到 <= l_min 的最大长度
    df_min = df_analysis[df_analysis['seq_len'] <= l_min]
    if len(df_min) == 0:
        # l_min 小于最小长度，累积值从 0 开始
        cumulative_mae_sum_min = 0.0
        cumulative_mse_sum_min = 0.0
        cumulative_count_min = 0
        cumulative_prob_min = 0.0
    else:
        row_min = df_min.iloc[-1]
        cumulative_mae_sum_min = row_min['cumulative_mae_sum']
        cumulative_mse_sum_min = row_min['cumulative_mse_sum']
        cumulative_count_min = row_min['cumulative_count']
        cumulative_prob_min = row_min['cumulative_prob']

    # 区间内的总误差 = 累积到 l_max 的总误差 - 累积到 l_min 的总误差
    interval_mae_sum = row_max['cumulative_mae_sum'] - cumulative_mae_sum_min
    interval_mse_sum = row_max['cumulative_mse_sum'] - cumulative_mse_sum_min

    # 区间内的样本数和概率
    interval_count = row_max['cumulative_count'] - cumulative_count_min
    interval_prob = row_max['cumulative_prob'] - cumulative_prob_min

    if interval_count == 0:  # 避免除零
        return {'mae': None, 'mse': None, 'count': 0, 'prob': 0.0}

    # 区间平均误差 = 区间总误差 / 区间样本数
    mae_interval = interval_mae_sum / interval_count
    mse_interval = interval_mse_sum / interval_count

    return {
        'mae': mae_interval,
        'mse': mse_interval,
        'count': int(interval_count),
        'prob': interval_prob
    }


def compute_interval_integral(seqlens, errors, interval):
    """
    计算单个区间的积分性能 E[error | l_min < L ≤ l_max] (直接计算版本)

    与 pd.cut 的左开右闭区间 (a, b] 保持一致

    Args:
        seqlens: 序列长度数组
        errors: 误差字典 {'abs': abs_errors, 'squared': squared_errors}
        interval: 区间元组 (l_min, l_max)，表示 (l_min, l_max]，左开右闭

    Returns:
        dict: {
            'mae': 区间内的平均绝对误差,
            'mse': 区间内的平均平方误差,
            'count': 区间内样本数,
            'prob': 区间内样本占比
        }
    """
    l_min, l_max = interval

    # 左开右闭区间: (l_min, l_max] 即 l_min < L ≤ l_max
    mask = (seqlens > l_min) & (seqlens <= l_max)
    count = mask.sum()

    if count == 0:
        return {'mae': None, 'mse': None, 'count': 0, 'prob': 0.0}

    # 区间内的性能
    mae = errors['abs'][mask].mean()
    mse = errors['squared'][mask].mean()
    prob = count / len(seqlens)

    return {
        'mae': mae,
        'mse': mse,
        'count': count,
        'prob': prob
    }


def compute_interval_metrics_bucket_method(df_meta, predictions, labels, interval):
    """
    使用分桶方法计算单个区间性能

    与 pd.cut 的左开右闭区间 (a, b] 保持一致

    Args:
        df_meta: 包含 'seq_len' 列的 DataFrame
        predictions: 预测值数组
        labels: 真实标签数组
        interval: 区间元组 (l_min, l_max)，表示 (l_min, l_max]，左开右闭

    Returns:
        dict: {'mae': ..., 'mse': ..., 'count': ..., 'prob': ...}
    """
    l_min, l_max = interval
    seqlens = df_meta['seq_len'].to_numpy()

    # 左开右闭区间: (l_min, l_max]
    mask = (seqlens > l_min) & (seqlens <= l_max)

    if not mask.any():
        return {'mae': None, 'mse': None, 'count': 0, 'prob': 0.0}

    pred_interval = predictions[mask]
    labels_interval = labels[mask]

    mae = np.abs(pred_interval - labels_interval).mean()
    mse = ((pred_interval - labels_interval) ** 2).mean()
    count = mask.sum()
    prob = count / len(predictions)

    return {
        'mae': mae,
        'mse': mse,
        'count': count,
        'prob': prob
    }


def compute_interval_metrics_integral_method(df_analysis, interval):
    """
    使用积分方法计算单个区间性能 (利用 analyze_model 的累积性能结果)

    与 pd.cut 的左开右闭区间 (a, b] 保持一致

    Args:
        df_analysis: analyze_model 返回的 DataFrame，包含 cumulative_mae/cumulative_mse
        interval: 区间元组 (l_min, l_max)，表示 (l_min, l_max]，左开右闭

    Returns:
        dict: {'mae': ..., 'mse': ..., 'count': ..., 'prob': ...}
    """
    return compute_interval_integral_from_cumulative(df_analysis, interval)


def verify_interval_methods_consistency(model='albert', interval=None):
    """
    验证分桶方法和积分方法在单个区间上的结果一致性

    对于同一个区间，两种方法应该给出相同的结果:
    - 分桶方法: 直接计算区间内样本的误差均值
    - 积分方法: 利用 analyze_model 的累积性能结果计算

    Args:
        model: 模型名称
        interval: 区间元组 (l_min, l_max)，表示 (l_min, l_max]，左开右闭
                  如果为 None 则使用默认区间 (39.0, 135.0]

    Returns:
        tuple: (bucket_result, integral_result, comparison_dict)
    """
    # 加载数据
    df = pl.read_parquet('data/processed/hf_saved/test.parquet')
    df_meta = df.select(pl.col('item_seq').list.len().alias('seq_len'))

    predictions = torch.load(f'final_results/{model}_output.pt', weights_only=False)
    labels = np.ones(len(predictions))

    # 如果没有提供 interval，使用默认区间
    if interval is None:
        seqlen_ser = df_meta['seq_len'].to_pandas()
        describe_ser = seqlen_ser.describe()
        Q1, Q2, Q3 = describe_ser[r'25% 50% 75%'.split()]
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        bins = [0, Q1, Q2, Q3, upper_bound, describe_ser['max']]
        # 默认使用中间区间 (Q1, Q2]
        interval = (bins[1], bins[2])

    # 方法1: 分桶方法 (直接计算)
    bucket_result = compute_interval_metrics_bucket_method(df_meta, predictions, labels, interval)

    # 方法2: 积分方法 (利用 analyze_model 的结果)
    # save_dir = Path('final_results/序列长度性能分析')
    # save_dir.mkdir(parents=True, exist_ok=True)
    df_analysis = analyze_model(model, df_meta)
    integral_result = compute_interval_metrics_integral_method(df_analysis, interval)

    # 对比结果
    comparison = {
        'model': model,
        'interval': f'({interval[0]}, {interval[1]}]',
        'mae_bucket': bucket_result['mae'],
        'mae_integral': integral_result['mae'],
        'mae_diff': abs(bucket_result['mae'] - integral_result['mae']) if bucket_result['mae'] is not None else None,
        'mse_bucket': bucket_result['mse'],
        'mse_integral': integral_result['mse'],
        'mse_diff': abs(bucket_result['mse'] - integral_result['mse']) if bucket_result['mse'] is not None else None,
        'count': bucket_result['count']
    }

    print(f"\n{'='*60}")
    print(f"模型 {model.upper()} - 单区间性能对比验证")
    print(f"{'='*60}")
    print(f"区间: {comparison['interval']}")
    print(f"样本数: {comparison['count']}")
    print(f"\n分桶方法: MAE={bucket_result['mae']:.6f}, MSE={bucket_result['mse']:.6f}")
    print(f"积分方法: MAE={integral_result['mae']:.6f}, MSE={integral_result['mse']:.6f}")
    print(f"\nMAE 差异: {comparison['mae_diff']:.10f}")
    print(f"MSE 差异: {comparison['mse_diff']:.10f}")

    # 验证一致性 (允许浮点数误差)
    tolerance = 1e-10
    is_consistent = comparison['mae_diff'] < tolerance and comparison['mse_diff'] < tolerance

    if is_consistent:
        print(f"✓ 验证通过: 两种方法结果一致")
    else:
        print(f"✗ 验证失败: 存在显著差异")

    return bucket_result, integral_result, comparison


if __name__ == '__main__':
    verify_interval_methods_consistency(interval=(39, 74))