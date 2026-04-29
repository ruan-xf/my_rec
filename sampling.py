"""
新的采样策略：类别和物品都会进行采样
使用 [0,1] 的分数表示模型正确预测用户偏好的程度

采样组合及分数：
- 真实类目 + 真实物品: 1.0
- <pad> + 真实物品: 1.0
- 真实类目/<pad> + 同类错误物品: 0.8
- 真实类目/<pad> + 不同类错误物品: 0.4
- 非真实类目 + 任意物品: 0.0
"""

import random
from pathlib import Path
from typing import Dict
import pandas as pd

import config


# 数据文件路径
DATA_DIR = Path('data/processed')
CATEGORY_ITEMS_FILE = DATA_DIR / 'category_items.parquet'
UNIQUE_ITEMS_FILE = DATA_DIR / 'unique_items.parquet'

pad_mark = config.special_tokens_map['pad_token']

df = pd.read_parquet(CATEGORY_ITEMS_FILE)
category_to_items = df.set_index('category_id')['item_id']
unique_items = pd.read_parquet(UNIQUE_ITEMS_FILE)
all_categories = category_to_items.index
all_items = unique_items.item_id.unique()


def prepare():
    """
    准备采样所需的数据文件：
    - category_items.parquet: 类目到物品的映射
    - unique_items.parquet: 唯一物品列表
    """
    from data_util import read_full

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = read_full()

    unique_items = (
        df['item_seq']
        .explode().struct.unnest()['category_id', 'item_id']
        .unique()
    )

    unique_items.write_parquet(UNIQUE_ITEMS_FILE)

    (
        unique_items
        .group_by('category_id')
        .agg('item_id')
        .write_parquet(CATEGORY_ITEMS_FILE)
    )

    print(f"Generated {UNIQUE_ITEMS_FILE} and {CATEGORY_ITEMS_FILE}")


def get_another_one(items, this_one):
    """从列表中随机选择一个不同于当前元素的元素"""
    if len(items) <= 1: return None
    return next(x for _ in iter(int, 1) if (x := random.choice(items)) != this_one)


def make_sample_result(item_id: int, category_id, label: float):
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
    if sample_item is None: return

    sample_category = pad_mark if is_pad else target_category

    return make_sample_result(sample_item, sample_category, 0.8)


def sample_diff_category_wrong_item(target_item: Dict, is_pad: bool = False):
    """
    真实类目/<pad> + 不同类别错误物品
    目标分数: 0.4
    """
    target_category = target_item['category_id']

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


def generate_train_samples(target_item: Dict):
    """
    生成训练样本，包含多种采样组合

    Args:
        target_item: 目标物品 {'item_id': int, 'category_id': int, ...}

    return: 所有的采样
    """
    sample_results = [
        sample_positive(target_item, False),
        sample_positive(target_item, True),
        sample_same_category_wrong_item(target_item, False),
        sample_same_category_wrong_item(target_item, True),
        sample_diff_category_wrong_item(target_item, False),
        sample_diff_category_wrong_item(target_item, True),
        sample_wrong_category_correct_item(target_item),
        sample_wrong_category_wrong_item(target_item)
    ]
    random.shuffle(sample_results)
    return sample_results


def generate_eval_sample(target_item: Dict):
    """
    生成验证/测试样本，只使用真实正样本
    目标分数: 1.0
    """
    yield sample_positive(target_item, is_pad=False)
    yield sample_positive(target_item, is_pad=True)
