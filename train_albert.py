from dataclasses import dataclass
import random
import wandb

import utils
import config

import common
from common import setup_experiment, create_ds_setting, build_trainer_params, collate_fn, FastDevRun
from modeling_albert import model_init


@dataclass
class AlbertTrainingArguments(config.MyDefaultTrainingArguments):
    output_dir: str = 'albert_rec'
    per_device_train_batch_size: int = 6
    per_device_eval_batch_size: int = 20
    gradient_accumulation_steps: int = 42
    max_steps: int = 20000
    # warmup_ratio: float = 0.1
    # learning_rate: float = 0.001
    warmup_steps: int = 1600
    learning_rate: float = 2e-4
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
    args, _ = setup_experiment(args)
    ds_setting = create_ds_setting()
    # for checkpoint resume
    # random.shuffle(ds_setting.splits['train'])
    # random.shuffle(ds_setting.splits['eval'])

    trainer_params = build_trainer_params(
        model_init(), args, ds_setting, collate_fn,
        early_stop_patience=3 * 6
    )
    trainer = utils.trainer_init(trainer_params, ds_setting)
    trainer.train()
    if wandb.run:
        wandb.run.finish()


# fast_dev_run = AlbertFastDevRun()
# # fast_dev_run.verbose = False
# fast_dev_run.delete_output = False

# fast_dev_run()

train()
