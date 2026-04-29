
import datetime
from functools import partial, reduce
import os
from pathlib import Path
import random
from typing import Callable
import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import OrdinalEncoder

from datasets import load_dataset, Dataset, IterableDatasetDict, IterableDataset

from sklearn.model_selection import train_test_split
import torch
import transformers

import config
import sampling


def convert_raw():
    pl.read_csv(
        '../data/UserBehavior.csv',
        has_header=False,
        new_columns=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'],
    ).write_parquet('../data/raw.parquet')

def save_full():
    begin_at = datetime.date(2017, 11, 25)
    end_at = datetime.date(2017, 12, 3)
    file_path = Path('data/processed/full.parquet')
    file_path.parent.mkdir(parents=True, exist_ok=True)
    (
        pl.scan_parquet('../data/raw.parquet')
        .with_columns(pl.from_epoch(pl.from_epoch("timestamp", time_unit="s"), time_unit='d').alias('date'))
        .filter(
            (begin_at <= pl.col('date')) & (pl.col('date') <= end_at)
        )
        .sort(['user_id', 'timestamp'])
        .group_by('user_id')
        .agg(
            pl.struct(config.seq_features).alias('item_seq'),
            pl.len().alias('item_seq_len'),
        )
        .collect(engine='gpu')
        .sample(fraction=1, shuffle=True)
    ).write_parquet('data/processed/full.parquet')


def read_full():
    return pl.read_parquet('data/processed/full.parquet')

def read_split_for_full():
    return pd.read_json('data/processed/split.json', typ='series')

def do_split():
    train_df = read_full()
    idx = np.arange(train_df.shape[0])

    train_idx, eval_idx = train_test_split(
        idx,
        test_size=0.3,
    )

    eval_idx, test_idx = train_test_split(
        eval_idx,
        test_size=0.7
    )


    split = pd.Series({
        'train': train_idx, 
        'eval': eval_idx,
        'test': test_idx
    })

    split.to_json('data/processed/split.json')


def save_test_sample():
    df = read_full()
    splits = read_split_for_full()
    for split_name, idx in splits.items():
        df[idx[:100]].write_parquet(f'data/processed/sample/{split_name}.parquet')


def read_sample(split_name: str):
    """读取 sample 数据"""
    return pl.read_parquet(f'data/processed/sample/{split_name}.parquet')


# 应该在划分出训练集后再进行
def prepare_tokenizers():
    from tokenizers import models, pre_tokenizers, trainers, Tokenizer

    df: pl.LazyFrame = (
        read_full()[read_split_for_full().train]
        .select(pl.col("item_seq").list.explode().struct.unnest())
        .lazy()
    )
    for col in config.seq_features:
        pl_col = pl.col(col)
        n_elems, elems_text = (
            df
            .select(pl_col.unique())
            .select(
                pl_col.len().alias('n_elems'),
                pl_col.cast(str).str.join(' ').alias('elems_text'),
            )
            .collect()
            .to_pandas().squeeze()
            # .to_series().to_list()
        )

        # text_list = (
        #     df
        #     .select(pl_col.cast(str).str.join(' '))
        #     .collect(engine='gpu')
        #     .to_series().to_list()
        # )


        tokenizer = Tokenizer(models.WordLevel())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.add_special_tokens(config.sorted_tokens)

        hf_tokenizer = transformers.AlbertTokenizerFast(
            tokenizer_object=tokenizer,
        )
        
        # 设置过大的vocab_size会kernel dead
        hf_tokenizer.train_new_from_iterator(
            [elems_text],
            len(hf_tokenizer.all_special_ids) + n_elems,
            # text_list,
            # int(1e16),
        ).save_pretrained(f'tokenizers/{col}')

def keep_only_the_most_recent(row: dict, max_history: int):
    item_seq = row['item_seq']
    item_seq = item_seq[-max_history:]
    n_items = len(item_seq)
    row.update({
        'item_seq': item_seq,
        'item_seq_len': n_items
    })
    return row

def get_max_history_setting_fn(max_history: int):
    return partial(keep_only_the_most_recent, max_history=max_history)


def compose(fn_list) -> Callable:
    fn_list = [f for f in fn_list if f is not None]
    if not fn_list: return
    return reduce(lambda f1, f2: lambda x: f2(f1(x)), fn_list)

class DatasetSetting:
    def __init__(self, per_eval_size: int, *, use_sample=False, process_fn: Callable=None):
        self.use_sample = use_sample
        if not use_sample:
            self.full_df = read_full()
            self.splits = read_split_for_full()
        self.per_eval_size = per_eval_size

        self.procss_fn = process_fn
        self.map_for_train, self.map_for_test = [process_fn], [process_fn]

        self.init_train_dataset()
        self.init_eval_dataset()
        self.init_test_dataset()
        self.eval_iter = None

    def init_train_dataset(self):
        self._train_dataset = IterableDataset.from_generator(
            lambda: self.generate_with_neg('train')
        )

    def init_eval_dataset(self):
        self.indefinite_eval_dataset = IterableDataset.from_generator(
            lambda: self.generate_with_neg('eval')
        ).repeat(None)
        
    def init_test_dataset(self):
        self._test_dataset = IterableDataset.from_generator(
            lambda: self.generate_with_neg('test')
        )
        
    @property
    def eval_dataset(self):
        if self.eval_iter is None: self.reset_eval_iter()
        return Dataset.from_dict(next(self.eval_iter)) #.map(None)

    @property
    def train_dataset(self):
        return self._train_dataset.map(compose(self.map_for_train))

    @property
    def test_dataset(self):
        return self._test_dataset.map(compose(self.map_for_test))

    def set_max_history_for_test(self, max_history: int):
        max_history_setting_fn = get_max_history_setting_fn(max_history)
        self.map_for_test = [max_history_setting_fn, self.process_fn]
        self.eval_iter = None

    def set_max_history_for_train(self, max_history: int):
        max_history_setting_fn = get_max_history_setting_fn(max_history)
        self.map_for_train = [max_history_setting_fn, self.process_fn]

    def reset_eval_iter(self):
        self.eval_iter = self.indefinite_eval_dataset.map(compose(self.map_for_test)).iter(self.per_eval_size)

    @staticmethod
    def generate_samples(item_seq, is_train):
        """
        使用新的采样策略生成样本

        Args:
            item_seq: 物品序列
            is_train: 是否为训练集

        Yields:
            格式化的样本字典，包含 item_seq 和 label
        """
        target_item = item_seq[-1]
        item_seq = pd.DataFrame(item_seq).astype(str).to_dict(orient='records')
        history_seq = item_seq[:-1]
        
        # if not is_train: yield {'item_seq': item_seq, 'label': 1.0}

        sample_results = []
        if is_train:
            # 训练时使用多种采样组合
            sample_results = sampling.generate_train_samples(target_item)
        else:
            # 验证/测试时只使用真实正样本
            sample_results = sampling.generate_eval_sample(target_item)

        for result in sample_results:
            if result is None: continue
            # 构建新的 item_seq，将采样结果作为最后一个物品
            new_item = config.get_a_empty()
            new_item['item_id'] = str(result['item'])
            new_item['category_id'] = str(result['category'])

            yield {
                'item_seq': history_seq + [new_item],
                'label': result['label']
            }

    def generate_with_neg(
        self,
        split_name: str,
    ):
        if self.use_sample:
            # 使用 sample 数据时，直接读取对应的 parquet 文件
            df: pl.DataFrame = read_sample(split_name)
        else:
            # 使用 full 数据时，使用索引切片
            df: pl.DataFrame = self.full_df[self.splits[split_name]]

        for item_seq, *_ in df.select('item_seq').iter_rows():
            yield from self.generate_samples(item_seq, split_name == 'train')

# save_full()
# do_split()
# prepare_tokenizers()



# ds = load_dataset(
#     'parquet',
#     data_files='data/processed/full.parquet',
#     cache_dir='data/cache',
#     split='train',
#     # streaming=True,
# )
# 
# ds = Dataset.from_parquet(
#     'data/processed/full.parquet',
#     split='train',
#     cache_dir='data/cache',
#     # streaming=True,
# )


# ds: IterableDataset


# lf = read_split_lazy('train')
# lf.head().collect()



# ds_setting = DatasetSetting(900, use_sample=True)
