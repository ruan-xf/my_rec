"""
模型和数据工具函数

包含数据采样、模型加载、获取注意力权重等函数。
这些函数依赖重的模型和数据模块，只在需要时导入。
"""

import pickle
from pathlib import Path
from typing import Optional

import polars as pl
import torch
import transformers

# 切换到项目根目录
import os
os.chdir('..')

import common
import data_util
import modeling_albert


def sample_item_seq(df: pl.DataFrame, seq_len: int) -> list:
    """从测试集中随机采样一个指定长度的 item_seq

    Args:
        seq_len: 目标序列长度
        seed: 随机种子（可选）

    Returns:
        item_seq 列表
    """
    filtered = df.filter(pl.col('item_seq_len') == seq_len)
    return filtered.sample(n=1)['item_seq'].first()

def load_test_data():
    return data_util.read_full()[data_util.read_split_for_full()['test']]

def model_init():
    return modeling_albert.AlbertRec.from_pretrained('checkpoints/albert_rec/checkpoint-3000')

def get_attention_weights(
    item_seq: list,
    *,
    model: modeling_albert.AlbertRec,
) -> torch.Tensor:
    """获取指定输入的注意力权重

    Args:
        item_seq: 物品序列
        model_path: 模型检查点路径
        sample_index: 使用第几个生成的样本（默认0）

    Returns:
        注意力权重张量，shape=(num_heads, seq_len, seq_len)
    """
    attention_module: transformers.models.albert.modeling_albert.AlbertAttention = (
        model.albert_classifier.albert.encoder.albert_layer_groups[0]
        .albert_layers[0].attention
    )

    sample = next(data_util.DatasetSetting.generate_samples(item_seq, False))

    processed_batch = common.collate_fn([sample])
    attention_weight = None

    def hook(_module, _input, output):
        nonlocal attention_weight
        attention_weight = output[1].clone()

    handle = attention_module.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        model(**processed_batch)

    handle.remove()
    return torch.squeeze(attention_weight)
