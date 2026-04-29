"""
按 cat_is_pad 分组分析模型性能并保存到 TensorBoard

对 cat_is_pad (序列最后一个 item 的 category_id 是否为 <pad>) 进行分组分析，
分别运行 analyze_model 并将结果保存到 TensorBoard。
"""

import polars as pl
import pandas as pd
import torch
import os
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from seqlen_distribution_analysis import analyze_model

# seqlen_distribution_analysis 导入时会执行 os.chdir('..')
# 确保工作目录正确
if not os.path.exists('data/processed/hf_saved/test.parquet'):
    os.chdir('..')


def compute_quartiles(seqlens):
    """
    计算四分位点分桶边界

    Args:
        seqlens: 序列长度数组

    Returns:
        bins: 分桶边界列表 [0, Q1, Q2, Q3, upper_bound, max]
    """
    seqlen_ser = pd.Series(seqlens)
    describe_ser = seqlen_ser.describe()

    Q1, Q2, Q3 = describe_ser[r'25% 50% 75%'.split()]
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    bins = [0, Q1, Q2, Q3, upper_bound, describe_ser['max']]
    return bins


def analyze_model_by_group(model: str, df_meta: pl.DataFrame):
    """
    按分组分析模型

    Args:
        model: 模型名称 ('albert', 'cnn', 'mlp')
        df_meta: 包含 seq_len 和 cat_is_pad 的 DataFrame

    Returns:
        dict: {None: df_global, True: df_true, False: df_false}
    """
    # 加载预测结果并拼接
    predictions = torch.load(f'final_results/{model}_output.pt', weights_only=False)
    df_meta = df_meta.with_columns(pl.Series('pred', predictions))

    results = {}

    # 全局分析
    results[None] = analyze_model(model, df_meta)

    # cat_is_pad=True
    df_true = df_meta.filter(pl.col('cat_is_pad') == True)
    results[True] = analyze_model(model, df_true)

    # cat_is_pad=False
    df_false = df_meta.filter(pl.col('cat_is_pad') == False)
    results[False] = analyze_model(model, df_false)

    return results


def save_results_to_csv(results, model: str, save_dir: Path):
    """
    保存分析结果到 CSV 文件

    Args:
        results: analyze_model_by_group 返回的结果字典
        model: 模型名称
        save_dir: 保存目录
    """
    model_dir = save_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)

    # 保存各分组结果
    group_mapping = {
        None: 'global',
        True: 'cat_is_pad_True',
        False: 'cat_is_pad_False'
    }

    for group_key, group_name in group_mapping.items():
        df_results = results[group_key]

        # 保存 CSV
        csv_path = model_dir / f'{group_name}.csv'
        df_results.to_csv(csv_path, index=False)
        print(f"  保存: {csv_path}")


def load_results_from_csv(model: str, save_dir: Path):
    """
    从 CSV 文件加载分析结果

    Args:
        model: 模型名称
        save_dir: 保存目录

    Returns:
        dict: {None: df_global, True: df_true, False: df_false}
    """
    model_dir = save_dir / model

    # 加载各分组结果
    results = {}
    group_mapping = {
        'global': None,
        'cat_is_pad_True': True,
        'cat_is_pad_False': False
    }

    for csv_name, group_key in group_mapping.items():
        csv_path = model_dir / f'{csv_name}.csv'
        df = pd.read_csv(csv_path)
        results[group_key] = df

    return results


def load_bins(save_dir: Path):
    """
    加载 bins 信息

    Args:
        save_dir: 保存目录

    Returns:
        dict: {None: bins_global, True: bins_true, False: bins_false}
    """
    bins_path = save_dir / 'bins.json'
    with open(bins_path, 'r') as f:
        bins_info = json.load(f)

    bins_by_group = {
        None: bins_info['global'],
        True: bins_info['cat_is_pad_True'],
        False: bins_info['cat_is_pad_False']
    }

    return bins_by_group


def save_to_tensorboard(df_results, writer):
    """
    保存结果到 TensorBoard

    Args:
        df_results: analyze_model 返回的 DataFrame
        writer: SummaryWriter 实例
    """
    # 设置 seq_len 为 index
    df_results.index = df_results.seq_len

    # 遍历每个列，记录标量曲线
    for col in df_results.columns:
        if col == 'seq_len':
            continue  # seq_len 已作为 index

        # 使用 group_name 作为前缀区分不同分组
        tag = f"test/{col}"
        for step, value in df_results[col].items():
            writer.add_scalar(tag, value, global_step=int(step))

    # # 记录四分位点信息
    # bins_text = f"{group_name} - 四分位点分桶: {bins}\n样本总数: {df_results['count'].sum()}"
    # writer.add_text(f'{group_name}/Quartiles_Info', bins_text, global_step=0)


def main():
    """主函数"""
    # 1. 加载数据
    df = pl.read_parquet('data/processed/hf_saved/test.parquet')
    df_meta = df.select(
        pl.col('item_seq').list.len().alias('seq_len'),
        pl.col('item_seq')
            .list.last()
            .struct.field('category_id')
            .eq('<pad>')
            .alias('cat_is_pad')
    )

    print(f"总样本数: {len(df_meta)}")
    print(f"cat_is_pad=True 样本数: {df_meta.filter(pl.col('cat_is_pad') == True).height}")
    print(f"cat_is_pad=False 样本数: {df_meta.filter(pl.col('cat_is_pad') == False).height}")

    # 验证分组样本总数等于全局样本数
    total_samples = df_meta.height
    true_samples = df_meta.filter(pl.col('cat_is_pad') == True).height
    false_samples = df_meta.filter(pl.col('cat_is_pad') == False).height
    assert total_samples == true_samples + false_samples, "分组样本数不一致"

    # 2. 计算每个分组的四分位点
    bins_by_group = {}
    # 全局
    bins_by_group[None] = compute_quartiles(df_meta['seq_len'].to_numpy())
    # cat_is_pad=True
    bins_by_group[True] = compute_quartiles(
        df_meta.filter(pl.col('cat_is_pad') == True)['seq_len'].to_numpy()
    )
    # cat_is_pad=False
    bins_by_group[False] = compute_quartiles(
        df_meta.filter(pl.col('cat_is_pad') == False)['seq_len'].to_numpy()
    )

    print("\n四分位点分桶:")
    print(f"  全局: {bins_by_group[None]}")
    print(f"  cat_is_pad=True: {bins_by_group[True]}")
    print(f"  cat_is_pad=False: {bins_by_group[False]}")

    # 3. 分析每个模型并保存结果
    models = ['albert', 'cnn', 'mlp']
    save_dir = Path('final_results/指标的tensorboard可视化')
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存 bins 信息（所有模型共享）
    bins_info = {
        'global': bins_by_group[None],
        'cat_is_pad_True': bins_by_group[True],
        'cat_is_pad_False': bins_by_group[False]
    }
    bins_path = save_dir / 'bins.json'
    with open(bins_path, 'w') as f:
        json.dump(bins_info, f, indent=2)
    print(f"\n保存 bins 信息到: {bins_path}")

    for model in models:
        print(f"\n分析模型: {model}")
        results = analyze_model_by_group(model, df_meta)

        # 保存结果到 CSV
        save_results_to_csv(results, model, save_dir)
        print(f"完成 {model} 分析")

    print(f"\n结果已保存到: {save_dir}")
    print(f"\n使用 load_results_from_csv() 和 load_bins() 加载结果并写入 TensorBoard")

if __name__ == '__main__':
#     main()

    # 将结果写入 TensorBoard
    models = ['albert', 'cnn', 'mlp']
    save_dir = Path('final_results/指标的tensorboard可视化')
    logdir = Path('final_results/tb_logs_2')

    group_name_mapping = {
        None: 'global',
        True: 'cat_is_pad_True',
        False: 'cat_is_pad_False'
    }

    for model in models:
        print(f"\n写入 {model} 到 TensorBoard")
        results = load_results_from_csv(model, save_dir)

        for group, df_results in results.items():
            group_name = group_name_mapping[group]
            writer = SummaryWriter(logdir / model / group_name)
            save_to_tensorboard(df_results, writer)
            writer.close()
            print(f"  完成: {model}/{group_name}")

    print(f"\nTensorBoard 日志已保存到: {logdir}")
    print(f"运行命令查看: tensorboard --logdir {logdir}")
