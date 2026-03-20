
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
            lambda: self.generate_with_neg('test', n_sample=2)
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


    def generate_with_neg(
        self,
        split_name: str,
        *,
        n_sample=5,
    ):
        if self.use_sample:
            # 使用 sample 数据时，直接读取对应的 parquet 文件
            df: pl.DataFrame = read_sample(split_name)
        else:
            # 使用 full 数据时，使用索引切片
            df: pl.DataFrame = self.full_df[self.splits[split_name]]

        df = (
            df
            .select(
                pl.col('item_seq').list.eval(
                    pl.element().struct.with_fields(*(
                        pl.field(col).cast(str)
                        for col in config.seq_features
                    ))
                )
            )
        )
        for item_seq, *_ in df.iter_rows():
            items = [int(elem['item_id']) for elem in item_seq]
            

            # 由于 int() 始终返回 0，而哨兵值为 1（永不匹配），所以 iter(int, 1) 会生成一个无限迭代器，不断产出 0
            def get_a_neg():
                # x = next(x for _ in iter(int, 1) if (x:=random.randint(80, 5162429)) not in items)
                x = next(x for _ in iter(int, 1) if (x:=random.randint(1, 5163070)) not in items)
                return x

            def get_a_sample(is_pos: bool):
                item = config.get_a_empty()
                item['item_id'] = str(items[-1] if is_pos else get_a_neg())
                return {
                    # 'user': user,
                    'item_seq': item_seq[:-1] + [item],
                    # 'item': item,
                    # 'item_seq_len': n_items,
                    'label': is_pos + 0
                }

            # n_sample > 1
            i_pos = random.randint(0, n_sample-1)
            for i in range(n_sample):
                yield get_a_sample(i == i_pos)



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
