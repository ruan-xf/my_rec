import pickle

import datasets
import polars as pl
import torch
import transformers
from pathlib import Path


import os
os.chdir('..')

import common
import data_util

sub_dir = Path('attention_analyze')

def sample_item_seq(seq_len: int):
    """从测试集中随机采样一个指定长度的 item_seq"""
    df = data_util.read_full()[data_util.read_split_for_full()['test']]
    return df.filter(pl.col('item_seq_len') == seq_len).sample(n=1)['item_seq'].first()

# 使用示例
# item_seq = sample_item_seq(74)
file_path = sub_dir / 'item_seq.pkl'
# with open(file_path, 'wb') as f:
#     pickle.dump(item_seq, f)

with open(file_path, 'rb') as f:
    item_seq = pickle.load(f)


samples = list(data_util.DatasetSetting.generate_samples(item_seq, False))

import modeling_albert
model = modeling_albert.AlbertRec.from_pretrained('checkpoints/albert_rec/checkpoint-3000')
attention_module: transformers.models.albert.modeling_albert.AlbertAttention = model.albert_classifier.albert.encoder.albert_layer_groups[0].albert_layers[0].attention

# 获取指定输入的注意力权重
def get_attention_weights(sample):
    """输入一个样本，返回第一层 attention 的注意力权重"""
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
    return torch.squeeze(attention_weight)  # torch.Size([8, 74, 74])


# 使用示例
attention_weight = get_attention_weights(samples[0])

attention_weight_file = sub_dir / 'attention_weight.pt'
torch.save(attention_weight, attention_weight_file)

attention_weight = torch.load(attention_weight_file)

