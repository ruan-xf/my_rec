from dataclasses import dataclass
import shutil
import transformers
import wandb

import utils
import config

from common import setup_experiment, create_ds_setting, build_trainer_params, collate_fn, FastDevRun
from modeling_albert import model_init


@dataclass
class AlbertTrainingArguments(config.MyDefaultTrainingArguments):
    output_dir: str = 'albert_rec'
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 30
    gradient_accumulation_steps: int = 32
    max_steps: int = 20000
    warmup_ratio: float = 0.1
    learning_rate: float = 0.001
    lr_scheduler_type: str = "constant_with_warmup"
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = logging_steps * 10
    eval_steps: int = logging_steps * 10
    


# 复用FastDevRun
class AlbertFastDevRun(FastDevRun):
    model_init = model_init
    args_class = AlbertTrainingArguments


def train():
    args = AlbertTrainingArguments()
    # args, _ = setup_experiment(args)
    ds_setting = create_ds_setting()

    trainer_params = build_trainer_params(
        model_init(), args, ds_setting, collate_fn,
        early_stop_patience=3 * 6
    )
    trainer = utils.trainer_init(trainer_params, ds_setting)
    trainer.train()
    if wandb.run:
        wandb.run.finish()


AlbertFastDevRun()()
# train()
