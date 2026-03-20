"""
功能测试代码：测试 collate_fn 和 model 的前向传播

使用方法：
    python test_collate_and_model.py
"""

import pandas as pd
import torch
import transformers
from transformers import TrainingArguments, PreTrainedModel, PretrainedConfig, Trainer


# from modeling import AlbertRec, RecConfig
import modeling
from data_util import DatasetSetting
import config

from modeling import seq_features, feature_encoders

# 1. 原样返回的collate_fn，用于了解处理的batch
def identity_collate_fn(batch):
    """
    原样返回batch，用于查看原始batch的结构
    """
    return batch


def get_batch_from_dataset(is_identity=True):
    """
    从实际数据集获取batch（封装data_util.py的代码）
    
    Args:
        collate_fn_type: 'identity' 或 'processed'，指定使用哪种collate_fn
    
    Returns:
        batch: 从数据集获取的batch
    """
    # 创建DemoModel
    class DemoModel(PreTrainedModel):
        def forward(self, behavior_type, category_id, item_id, attention_mask, token_type_ids, labels, item_seq=None, **kwargs):
            pass
    
    demo_model = DemoModel(PretrainedConfig())
    
    # 创建TrainingArguments
    args = TrainingArguments(
        per_device_eval_batch_size=2,
        eval_strategy='steps',
        eval_steps=0.3,
        logging_steps=0.3,
        max_steps=20,
        save_strategy='no',
        report_to='none'
    )
    
    # 创建DatasetSetting
    ds_setting = DatasetSetting(900, use_sample=True)
    
    # 选择collate_fn
    if is_identity:
        collate_fn_to_use = identity_collate_fn
    else:
        collate_fn_to_use = modeling.collate_fn
    
    # 创建Trainer和DataLoader
    trainer = Trainer(
        demo_model,
        args=args,
        eval_dataset=ds_setting.eval_dataset,
        data_collator=collate_fn_to_use,
    )
    
    dataloader = trainer.get_eval_dataloader()
    batch = next(iter(dataloader))
    
    return batch

# get_batch_from_dataset(True)
# [{'item_seq': [{'behavior_type': 'pv',
#     'category_id': '3323023',
#     'item_id': '4071389'},
#    {'behavior_type': 'pv', 'category_id': '4193511', 'item_id': '1340579'},
#    {'behavior_type': 'pv', 'category_id': '1248986', 'item_id': '1790164'},
#    {'behavior_type': 'pv', 'category_id': '3597057', 'item_id': '1826838'},
#    {'behavior_type': 'pv', 'category_id': '135038', 'item_id': '4602800'},
#    {'behavior_type': 'pv', 'category_id': '4445129', 'item_id': '756621'},
#    {'behavior_type': 'pv', 'category_id': '3284512', 'item_id': '1361723'},
#    {'behavior_type': 'pv', 'category_id': '3284512', 'item_id': '2019994'},
#    {'behavior_type': 'pv', 'category_id': '153309', 'item_id': '1743327'},
#    {'behavior_type': '<pad>', 'category_id': '<pad>', 'item_id': '2276118'}],
#   'label': 1},
#  {'item_seq': [{'behavior_type': 'pv',
#     'category_id': '3323023',
#     'item_id': '4071389'},
#    {'behavior_type': 'pv', 'category_id': '4193511', 'item_id': '1340579'},
#    {'behavior_type': 'pv', 'category_id': '1248986', 'item_id': '1790164'},
#    {'behavior_type': 'pv', 'category_id': '3597057', 'item_id': '1826838'},
#    {'behavior_type': 'pv', 'category_id': '135038', 'item_id': '4602800'},
#    {'behavior_type': 'pv', 'category_id': '4445129', 'item_id': '756621'},
#    {'behavior_type': 'pv', 'category_id': '3284512', 'item_id': '1361723'},
#    {'behavior_type': 'pv', 'category_id': '3284512', 'item_id': '2019994'},
#    {'behavior_type': 'pv', 'category_id': '153309', 'item_id': '1743327'},
#    {'behavior_type': '<pad>', 'category_id': '<pad>', 'item_id': '209806'}],
#   'label': 0}]

# get_batch_from_dataset(False)
# {'behavior_type': tensor([[8, 8, 8, 8, 8, 8, 8, 8, 8, 0],
#          [8, 8, 8, 8, 8, 8, 8, 8, 8, 0]]),
#  'category_id': tensor([[4675, 6442,  487, 5233,  671, 6947, 4594, 4594, 1055,    0],
#          [4675, 6442,  487, 5233,  671, 6947, 4594, 4594, 1055,    0]]),
#  'item_id': tensor([[2414391,  267490,  620711,  649490, 2831990, 3460913,  284047,  801278,
#                 1, 1002840],
#          [2414391,  267490,  620711,  649490, 2831990, 3460913,  284047,  801278,
#                 1,  862600]]),
#  'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]),
#  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
#  'labels': tensor([1, 0])}

def test_model_forward():
    processed_batch = get_batch_from_dataset(False)

    # 创建模型并测试
    model = modeling.model_init()
    model.eval()  # 设置为评估模式
    
    with torch.no_grad():
        output = model(**processed_batch)


    return output

