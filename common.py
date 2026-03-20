
from datetime import datetime
import shutil
from typing import Type

import pandas as pd
import torch
import transformers
from dataclasses import dataclass
from pathlib import Path
import os

import config as global_config
import utils


# ==================== 数据处理相关 ====================

seq_features = global_config.seq_features

feature_encoders: dict[str, transformers.AlbertTokenizerFast] = {
    col: transformers.AutoTokenizer.from_pretrained(
        f'tokenizers/{col}',
        truncation_side='left',
        **{"model_max_length": 512},
    )
    for col in seq_features
}


def collate_fn(batch):
    df = pd.DataFrame(batch)
    atten_dic = {}
    features = {}
    for col in seq_features:
        out = feature_encoders[col](
            [' '.join(d[col] for d in seq) for seq in df.item_seq],
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        features[col] = out.input_ids
        if col == 'item_id': atten_dic = out
        
    atten_dic.pop('input_ids')
    atten_dic['token_type_ids'][:, -1] = 1
    return {
        **features,
        **atten_dic,
        'labels': torch.tensor(df.label.tolist()),
    }

# ==================== 训练配置相关 ====================

@dataclass
class DefaultTrainingArguments(global_config.MyDefaultTrainingArguments):
    """通用训练参数配置基类"""
    pass




class FeatureEmbeddingMixin:
    """
    特征embedding混合类，供MLP、CNN等模型复用
    """
    def _init_feature_embeddings(self, config):
        """
        初始化特征embedding层

        Args:
            config: 包含 feature_vocab_sizes, embedding_size, pad_token_id 的配置对象
        """
        self.feature_embeddings = torch.nn.ModuleList()
        for col in seq_features:
            vocab_size = config.feature_vocab_sizes[col]
            self.feature_embeddings.append(
                torch.nn.Embedding(
                    vocab_size,
                    config.embedding_size // len(seq_features),
                    padding_idx=config.pad_token_id
                )
            )

    def _concat_feature_embeddings(self, behavior_type, category_id, item_id):
        """
        拼接特征embedding

        Returns:
            concated: shape (batch_size, seq_len, embedding_size)
        """
        return torch.cat([
            embeddings(data)
            for embeddings, data
            in zip(self.feature_embeddings, (behavior_type, category_id, item_id))
        ], dim=-1)


# ==================== 训练流程通用函数 ====================

def setup_experiment(args):
    """
    配置实验：
    目前为正式的多模型比较前的调参尝试

    Returns: (args, project_name)
    """
    args.run_name = datetime.now().strftime('%m%d-%H%M')
    project_name = Path(args.output_dir).name
    os.environ['WANDB_PROJECT'] = project_name


    # 1. 确定项目名和运行名，创建目录
    log_root = 'tb_logs'
    run_name = args.run_name
    run_log_dir = Path(log_root) / project_name / run_name
    # self.run_log_dir.mkdir(parents=True, exist_ok=True)

    # 2. 更新 args.logging_dir，让 transformers 直接写 tb_logs
    args.logging_dir = run_log_dir.as_posix()

    return args, project_name



def create_ds_setting(eval_size: int = 48*80, test_size: int = 48*300):
    """
    创建 DatasetSetting

    Args:
        eval_size: 评估集大小
        test_size: 测试集大小

    Returns: ds_setting
    """
    ds_setting = utils.DatasetSetting(eval_size)
    ds_setting._test_dataset = ds_setting._test_dataset.take(test_size)
    return ds_setting


def build_trainer_params(model, args, ds_setting, collate_fn, early_stop_patience=None):
    """
    构建 Trainer 参数字典

    Args:
        model: 模型实例或model_init函数
        args: TrainingArguments
        ds_setting: DatasetSetting
        collate_fn: 数据整理函数
        early_stop_patience: 早停patience（如果为None则不修改）

    Returns: trainer_params
    """
    trainer_params = dict(
        **utils.get_Trainer_common_params(model, args, ds_setting),
        data_collator=collate_fn,
    )
    if early_stop_patience is not None:
        early_stop_callback = trainer_params['callbacks'][0]
        early_stop_callback.early_stopping_patience = early_stop_patience
    return trainer_params

class FastDevRun:
    # 子类直接赋值原来的全局变量即可
    model_init = None  # 原模型初始化函数
    args_class: Type[transformers.TrainingArguments] = None  # 原训练参数类

    def __init__(self):
        self.eval_size = 20
        self.test_size = 20
        self.early_stop_patience = 3

        self.train_args = self.args_class(
            auto_find_batch_size=True,
            report_to='none',
            logging_steps=1,
            eval_steps=1,
            save_steps=1,
        )
        self.verbose = True
        self.delete_output = True


    def __call__(self):
        """执行fast_dev_run流程
        """
        # 小样本数据集
        ds_setting = create_ds_setting(eval_size=self.eval_size, test_size=self.test_size)
        # 构建训练器
        trainer_params = build_trainer_params(
            self.__class__.model_init(), self.train_args, ds_setting, collate_fn,
            early_stop_patience=self.early_stop_patience
        )
        trainer = utils.trainer_init(trainer_params, ds_setting)
        # 执行测试
        if self.verbose:
            transformers.logging.set_verbosity_debug()

        utils.trainer_start(trainer, ds_setting)

        if self.delete_output:
            shutil.rmtree(self.train_args.output_dir)

        if self.verbose:
            transformers.logging.set_verbosity_warning()
