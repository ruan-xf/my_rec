"""
新的采样策略：用于测试各种采样方法
"""
import random
from typing import Dict
import pandas as pd


import sys
sys.path.append('..')

import config


# 加载类目到物品的映射
df = pd.read_parquet('category_items.parquet')

category_to_items = df.set_index('category_id')['item_id']
unique_items = pd.read_parquet('unique_items.parquet')
all_categories = unique_items.category_id.unique()
all_items = unique_items.item_id.unique()


pad_mark = config.special_tokens_map['pad_token']

def get_another_one(items, this_one):
    assert len(items) > 1
    return next(x for _ in iter(int, 1) if (x:=random.choice(items)) != this_one)

def make_sample_result(item_id: int, category_id: int, label: float):
    """
    封装采样结果为统一格式
    返回: {'item': item_id, 'category': category_id, 'label': label}
    """
    return {
        'item': item_id,
        'category': category_id,
        'label': label
    }

def sample_positive(target_item: Dict, is_pad: bool = False):
    """
    真实类目/<pad> + 真实物品
    目标分数: 1.0
    """
    sample_item = target_item['item_id']
    sample_category = pad_mark if is_pad else target_item['category_id']

    return make_sample_result(sample_item, sample_category, 1.0)

def sample_same_category_wrong_item(target_item: Dict, is_pad: bool = False):
    """
    真实类目/<pad> + 同类错误物品
    目标分数: 0.8
    """
    target_category = target_item['category_id']

    sample_item = get_another_one(category_to_items[target_category], target_item['item_id'])
    sample_category = pad_mark if is_pad else target_category

    return make_sample_result(sample_item, sample_category, 0.8)


def sample_diff_category_wrong_item(target_item: Dict, is_pad: bool = False):
    """
    真实类目/<pad> + 不同类别错误物品
    目标分数: 0.4
    """
    target_category = target_item['category_id']

    # 从其他类目中收集物品
    other_category = get_another_one(all_categories, target_category)

    sample_item = random.choice(category_to_items[other_category])
    sample_category = pad_mark if is_pad else target_category

    return make_sample_result(sample_item, sample_category, 0.4)


def sample_wrong_category_correct_item(target_item: Dict):
    """
    非真实类目 + 真实物品
    目标分数: 0.0
    """
    target_category = target_item['category_id']

    sample_item = target_item['item_id']
    sample_category = get_another_one(all_categories, target_category)

    return make_sample_result(sample_item, sample_category, 0.0)


def sample_wrong_category_wrong_item(target_item: Dict):
    """
    非真实类目 + 错误物品
    目标分数: 0.0
    """
    sample_item = get_another_one(all_items, target_item['item_id'])
    sample_category = get_another_one(all_categories, target_item['category_id'])

    return make_sample_result(sample_item, sample_category, 0.0)
