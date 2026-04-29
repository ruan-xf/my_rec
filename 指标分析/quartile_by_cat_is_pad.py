"""
对 cat_is_pad 分组进行四分位点分桶分析

与 analyze_utils.py 中的逻辑相同,但分别对 cat_is_pad=True 和 False 的数据进行分桶
"""

import polars as pl
import pandas as pd
import torch
from pathlib import Path

import os
os.chdir('..')

import utils

# 加载数据
df = pl.read_parquet('data/processed/hf_saved/test.parquet')
df_meta = df.select(
    pl.col('item_seq').list.len().alias('seq_len'),
    pl.col('item_seq')
        .list.last()
        .struct.field('category_id')
        .eq('<pad>')
        .alias('cat_is_pad')
)

def load_model_output(model):
    output = torch.load(f'final_results/{model}_output.pt', weights_only=False)
    return df_meta.with_columns(pl.Series('pred', output))

def compute_grouped_metrics(df: pl.DataFrame, col):
    return df.to_pandas().groupby(col).apply(
        lambda g: pd.Series(utils.compute_metrics((g.pred.to_numpy(), torch.ones(len(g)))))
    )

def save_result_csv(results, save_path):
    num_cols = len(results[0])
    result_df = pd.DataFrame(results).set_index(list(range(num_cols-1)))
    pd.DataFrame(result_df.squeeze().to_list(), result_df.index).to_csv(save_path)

output = torch.load('final_results/cnn_output.pt', weights_only=False)
df_output = df_meta.with_columns(pl.Series('pred', output))


results = []

# 分组: cat_is_pad=True 和 cat_is_pad=False
for cat_is_pad in [True, False]:
    print(f"  处理 cat_is_pad={cat_is_pad}")

    # 筛选当前分组的数据
    df_group = df_output.filter(pl.col('cat_is_pad') == cat_is_pad)

    # 计算该分组的四分位点
    seqlen_ser = df_group['seq_len'].to_pandas()
    describe_ser = seqlen_ser.describe()

    Q1, Q2, Q3 = describe_ser[r'25% 50% 75%'.split()]
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    bins = [0, Q1, Q2, Q3, upper_bound, describe_ser['max']]

    print(f"    四分位点: {bins}")

    # 分桶
    bucket_ser = pd.cut(seqlen_ser, bins=bins)
    buckets = bucket_ser.dtype.categories

    bucket_ser_pl = pl.Series(bucket_ser).cast(str)
    df_group_with_bucket = df_group.with_columns(bucket_ser_pl.alias('bucket'))

    # 计算每个桶的指标
    metrics_df = compute_grouped_metrics(df_group_with_bucket, 'bucket')

    results.append((cat_is_pad, metrics_df))

results