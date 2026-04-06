"""
功能测试代码：测试 collate_fn 和 model 的前向传播

使用方法：
    python test_collate_and_model.py
"""

import pandas as pd
import torch
import transformers
from transformers import TrainingArguments, PreTrainedModel, PretrainedConfig, Trainer


import common
import utils
import config

import modeling_albert, modeling_cnn, modeling_mlp


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
    ds_setting = utils.DatasetSetting(900, use_sample=True)
    
    # 选择collate_fn
    if is_identity:
        collate_fn_to_use = identity_collate_fn
    else:
        collate_fn_to_use = common.collate_fn
    
    # 创建Trainer和DataLoader
    trainer = Trainer(
        demo_model,
        args=args,
        eval_dataset=ds_setting.train_dataset,
        data_collator=collate_fn_to_use,
    )
    
    dataloader = trainer.get_eval_dataloader()
    batch = next(iter(dataloader))
    
    return batch

def test_model_forward(model):
    processed_batch = get_batch_from_dataset(False)

    # 创建模型并测试
    model.eval()  # 设置为评估模式
    
    with torch.no_grad():
        output = model(**processed_batch)


    return output



# get_batch_from_dataset(True)
# get_batch_from_dataset(False)
# test_model_forward(modeling_mlp.model_init())
# test_model_forward(modeling_cnn.model_init())
# test_model_forward(modeling_albert.model_init())

# {'loss': tensor(0.3383), 'logits': tensor([-0.0223, -0.0214])}

# SequenceClassifierOutput(loss=tensor(0.4532), logits=tensor([[-0.1409],
#         [-0.1451]]), hidden_states=None, attentions=None)