from dataclasses import dataclass
import shutil
import transformers

import common
import utils
import config

from common import create_ds_setting_parquet, build_trainer_params, collate_fn, setup_experiment
from modeling_mlp import model_init


@dataclass
class MLPTrainingArguments(config.MyDefaultTrainingArguments):
    output_dir: str = 'mlp_rec'
    per_device_train_batch_size: int = 256
    per_device_eval_batch_size: int = 512
    gradient_accumulation_steps: int = 1
    max_steps: int = 20000
    # warmup_ratio: float = 0.1
    learning_rate: float = 0.001
    lr_scheduler_type: str = "constant"
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = logging_steps * 10
    eval_steps: int = logging_steps * 10


def train():
    args = MLPTrainingArguments()
    args, _ = setup_experiment(args)
    ds_setting = create_ds_setting_parquet()

    trainer_params = build_trainer_params(
        model_init(), args, ds_setting, collate_fn,
        early_stop_patience=3 * 6
    )
    trainer = utils.trainer_init(trainer_params, ds_setting)
    trainer.train()


class MLPFastDevRun(common.FastDevRun):
    model_init = model_init
    args_class = MLPTrainingArguments


# MLPFastDevRun()()


# fast_dev_run = MLPFastDevRun()
# fast_dev_run.verbose = False
# fast_dev_run.delete_output = False

# fast_dev_run()

train()
