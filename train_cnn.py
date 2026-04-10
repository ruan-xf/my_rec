from dataclasses import dataclass, field
import shutil
import transformers

import utils
import config

from common import setup_experiment, create_ds_setting_parquet, build_trainer_params, collate_fn, FastDevRun
from modeling_cnn import model_init
# from modeling_cnn import CNNModel, CNNConfig
# def model_init():
#     return CNNModel(CNNConfig(
#         kernel_size=6
#     ))


@dataclass
class CNNTrainingArguments(config.MyDefaultTrainingArguments):
    output_dir: str = 'cnn_rec'
    per_device_train_batch_size: int = 256
    per_device_eval_batch_size: int = 512
    gradient_accumulation_steps: int = 1
    max_steps: int = 20000
    # warmup_ratio: float = 0.1
    # warmup_steps: int = 800
    learning_rate: float = 5e-3
    lr_scheduler_type: str = "greedy"
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {
        'patience': 2,
        'min_lr': 1e-5,
        'max_lr': 0.1,
        'factor': 0.95,
        # 'smooth': True,
        # 'window_size': 50,
        # 'warmup': 3,
    })
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = logging_steps * 10
    eval_steps: int = logging_steps * 5


# 复用FastDevRun
class CNNFastDevRun(FastDevRun):
    model_init = model_init
    args_class = CNNTrainingArguments


def train():
    args = CNNTrainingArguments()
    args, _ = setup_experiment(args)
    ds_setting = create_ds_setting_parquet()

    trainer_params = build_trainer_params(
        model_init(), args, ds_setting, collate_fn,
        early_stop_patience=3 * 6
    )
    trainer = utils.trainer_init(trainer_params, ds_setting)
    trainer.train()


# CNNFastDevRun()()
train()
