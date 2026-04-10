from dataclasses import dataclass, field
import random

import utils
import config

import common
from common import setup_experiment, create_ds_setting_parquet, build_trainer_params, collate_fn, FastDevRun

# from modeling_albert import model_init
from modeling_albert import AlbertRec, RecConfig
def model_init():
    return AlbertRec(RecConfig(
        num_hidden_layers=1,
        dropout=0,
        embedding_weight_decay=1e-5,
    ))


@dataclass
class AlbertTrainingArguments(config.MyDefaultTrainingArguments):
    output_dir: str = 'albert_rec'
    per_device_train_batch_size: int = 6
    per_device_eval_batch_size: int = 20
    gradient_accumulation_steps: int = 42
    max_steps: int = 20000
    # warmup_ratio: float = 0.1
    # learning_rate: float = 0.001
    # warmup_steps: int = 800
    # learning_rate: float = 1e-3
    # lr_scheduler_type: str = "constant_with_warmup"
    learning_rate: float = 5e-4
    lr_scheduler_type: str = "greedy"
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {
        'patience': 2,
        'min_lr': 1e-6,
        'max_lr': 1e-2,
        'factor': 0.95,
        # 'smooth': True,
        # 'window_size': 50,
        # 'warmup': 1,
    })
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = logging_steps * 10
    eval_steps: int = logging_steps * 5
    


# 复用FastDevRun
class AlbertFastDevRun(FastDevRun):
    model_init = model_init
    args_class = AlbertTrainingArguments


def train():
    args = AlbertTrainingArguments()
    args, _ = setup_experiment(args)
    ds_setting = create_ds_setting_parquet()
    # for checkpoint resume
    # random.shuffle(ds_setting.splits['train'])
    # random.shuffle(ds_setting.splits['eval'])

    trainer_params = build_trainer_params(
        model_init(), args, ds_setting, collate_fn,
        early_stop_patience=3 * 6
    )
    trainer = utils.trainer_init(trainer_params, ds_setting)
    trainer.train()


# AlbertFastDevRun()()
# fast_dev_run = AlbertFastDevRun()
# # fast_dev_run.verbose = False
# fast_dev_run.delete_output = False

# fast_dev_run()

train()
