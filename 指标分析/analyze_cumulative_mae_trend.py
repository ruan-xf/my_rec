"""
分析cumulative_mae趋势与区间mae的关系

目的：获取不同分组的区间mae数据，验证cumulative_mae趋势变化
- cat_is_pad=False: 区间 0-100, 100-200, 200-300, 300-400
- cat_is_pad=True: 0-100内间隔10，之后使用已有区间结果
"""

from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import torch

from seqlen_distribution_analysis import compute_interval_metrics_bucket_method


def load_test_data_with_padding_info():
    """加载测试数据，包含padding信息"""
    df = pl.read_parquet('data/processed/hf_saved/test.parquet')
    df_meta = df.select([
        pl.col('item_seq').list.len().alias('seq_len'),
        pl.col('item_seq')
            .list.last()
            .struct.field('category_id')
            .eq('<pad>')
            .alias('cat_is_pad')
    ])
    return df_meta


def load_model_output(model: str):
    """加载模型预测结果"""
    output = torch.load(f'final_results/{model}_output.pt', weights_only=False)
    return output


def analyze_cat_is_pad_false(model: str, df_meta: pl.DataFrame, predictions: np.ndarray):
    """
    分析cat_is_pad=False的区间mae

    按照README第58-59行要求：
    - 区间：0-100, 100-200, 200-300, 300-400
    """
    # 筛选cat_is_pad=False的样本
    mask = df_meta['cat_is_pad'].to_numpy() == False
    preds = predictions[mask]
    labels = np.ones(len(preds))

    # 定义区间
    intervals = [(0, 100), (100, 200), (200, 300), (300, 400)]

    # 计算每个区间的mae
    interval_results = []
    for l_min, l_max in intervals:
        result = compute_interval_metrics_bucket_method(
            df_meta.filter(mask),
            preds,
            labels,
            (l_min, l_max)
        )
        interval_results.append({
            'interval': f'({l_min}, {l_max}]',
            'mae': result['mae']
        })

    return pd.DataFrame(interval_results)


def analyze_cat_is_pad_true(model: str, df_meta: pl.DataFrame, predictions: np.ndarray):
    """
    分析cat_is_pad=True的区间mae

    按照README第64-68行要求：
    - 0-100内间隔10
    """
    # 筛选cat_is_pad=True的样本
    mask = df_meta['cat_is_pad'].to_numpy() == True
    preds = predictions[mask]
    labels = np.ones(len(preds))

    # 定义区间：0-100内间隔10
    intervals_0_100 = [(i*10, (i+1)*10) for i in range(10)]

    # 计算每个小区间的mae
    interval_results = []
    for l_min, l_max in intervals_0_100:
        result = compute_interval_metrics_bucket_method(
            df_meta.filter(mask),
            preds,
            labels,
            (l_min, l_max)
        )
        interval_results.append({
            'interval': f'({l_min}, {l_max}]',
            'mae': result['mae']
        })

    return pd.DataFrame(interval_results)


def main():
    """分析所有模型"""
    # 加载数据
    df_meta = load_test_data_with_padding_info()
    model = 'cnn'
    predictions = load_model_output(model)

    # cat_is_pad=False分析
    results_false = analyze_cat_is_pad_false(model, df_meta, predictions)

    # cat_is_pad=True分析
    results_true = analyze_cat_is_pad_true(model, df_meta, predictions)

    results = (
        (False, results_false),
        (True, results_true)
    )

    return results


main()
