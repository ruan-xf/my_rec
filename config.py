from dataclasses import dataclass, field
from transformers import AlbertConfig, TrainingArguments


# 记得写上类型，否则不能覆盖
@dataclass
class MyDefaultTrainingArguments(TrainingArguments):
    auto_find_batch_size: bool = True
    max_steps: int =100000
    eval_strategy: str = 'steps'
    metric_for_best_model: str = 'eval_roc_auc'
    load_best_model_at_end: bool = True
    greater_is_better: bool = True
    save_total_limit: int = 2
    report_to: str = 'tensorboard'

    def __post_init__(self):
        super().__post_init__()
        if hasattr(self, 'output_dir') and not self.output_dir.startswith('checkpoints/'):
            self.output_dir = f'checkpoints/{self.output_dir}'


seq_features = 'behavior_type category_id item_id'.split()

# 特征词汇表大小配置
feature_vocab_sizes = {'behavior_type': 9, 'category_id': 9277, 'item_id': 3652291}

# item_id 总范围（数据范围需要进一步确认）
item_id_range = (1, 5163070)

added_tokens_encoder = {'<pad>': 0, '<unk>': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}

sorted_tokens = list(added_tokens_encoder.keys())

special_tokens_map = {'bos_token': '[CLS]',
 'eos_token': '[SEP]',
 'unk_token': '<unk>',
 'sep_token': '[SEP]',
 'pad_token': '<pad>',
 'cls_token': '[CLS]',
 'mask_token': '[MASK]'}


def get_a_empty():
    return dict.fromkeys(seq_features, special_tokens_map['pad_token'])


# import pandas as pd


# token_df = pd.merge(
#     pd.DataFrame.from_dict(
#         special_tokens_map,
#         orient='index',
#         columns=['token']).reset_index(names='token_type'),
#     pd.DataFrame.from_dict(
#         added_tokens_encoder,
#         orient='index', columns=['encode_id']
#     ).reset_index(names='token'),
# )