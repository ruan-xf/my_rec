
import dataclasses
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable
import evaluate
from scipy.special import softmax
from torch import nn

import torch
from transformers import (
    Trainer, TrainingArguments,
    EarlyStoppingCallback, TrainerCallback,
    logging,
)

# logging.enable_progress_bar()  
# logging.set_verbosity_info()

import subprocess
import webbrowser
from tqdm.auto import tqdm
import transformers
import wandb

from data_util import DatasetSetting as _DatasetSetting
# from data_util import DatasetSetting
from config import MyDefaultTrainingArguments

from datasets import Dataset, IterableDataset


class DatasetSetting:
    """使用parquet文件的DatasetSetting"""
    def __init__(self, per_eval_size, *args, **kwargs):
        self.train_dataset = Dataset.from_parquet(
            'data/processed/hf_saved/train.parquet',
            cache_dir='data/cache',
        )
        self._eval_dataset = IterableDataset.from_parquet(
            'data/processed/hf_saved/eval.parquet'
        ).repeat(None)
        self.eval_iter = None
        self.per_eval_size = per_eval_size

    @property
    def eval_dataset(self):
        if self.eval_iter is None:
            self.reset_eval_iter()
        return Dataset.from_dict(next(self.eval_iter))

    @property
    def test_dataset(self):
        return self.eval_dataset

    def reset_eval_iter(self):
        self.eval_iter = self._eval_dataset.iter(self.per_eval_size)


roc_auc_score = evaluate.load("roc_auc")


# logits = [[ 1.4070835, -1.4878857],
# [ 1.390159,  -1.4613104]]
# softmax(logits, axis=1)
# array([[0.94759719, 0.05240281],
#        [0.94539459, 0.05460541]])
def convert_to_single_output(logits):
    # 将logits转换为正类的概率分数
    # 应用softmax函数得到概率分布
    probabilities = softmax(logits, axis=1)
    
    # 提取正类的概率（索引为1）
    return probabilities[:, 1]
    

# eval_roc_auc
def compute_metrics(eval_pred, do_convert: bool):
    # 获取预测的logits和真实标签
    logits, labels = eval_pred
    pred_scores = convert_to_single_output(logits) if do_convert else logits
    
    # 构建metric输入
    metric_inputs = {
        "prediction_scores": pred_scores,
        "references": labels
    }
    
    return roc_auc_score.compute(**metric_inputs)

class TestTqdmCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None
        self.eval_batch_size: int = None

    def start(self):
        msg = f'***** Running Evaluation *****\n  Batch size = {self.eval_batch_size}'
        print(msg)
        self.pbar = tqdm()
    
    def step(self):
        if self.pbar is not None:
            self.pbar.update(1)
    
    def end(self):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
    
    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if transformers.trainer_utils.has_length(eval_dataloader): return
        self.eval_batch_size = args.eval_batch_size
        if self.pbar is None: self.start()
        self.step()

    def on_evaluate(self, args, state, control, **kwargs):
        self.end()

class EvalSlidingCallback(TrainerCallback):
    def __init__(self, trainer: Trainer, ds_setting: DatasetSetting, patience: int=3):
        self.trainer = trainer
        self.ds_setting = ds_setting
        self.in_trainloop = False
        self.counter = 0
        self.patience = patience

    def on_train_begin(self, args, state, control, **kwargs):
        self.in_trainloop = True

    def on_train_end(self, args, state, control, **kwargs):
        self.in_trainloop = False
        self.ds_setting.reset_eval_iter()

    def on_evaluate(self, args, state, control, **kwargs):
        if not self.in_trainloop: return
        self.counter += 1
        if self.counter == self.patience:
            self.trainer.eval_dataset = self.ds_setting.eval_dataset
            self.counter = 0


class TensorBoardLauncherCallback(TrainerCallback):
    """
    整合 TensorBoard 管理功能：
    1. 根据 args.report_to 自动检测是否启用
    2. 自动启动 TensorBoard 服务并打开浏览器
    3. 创建独立的日志目录结构 tb_logs/{project}/{run_name}
    4. 从 checkpoint 复制 config 文件（只复制一次）
    5. 同步复制 checkpoints 到日志目录
    """
    def __init__(self, log_root: str = "tb_logs", port: int = 6006, auto_launch: bool = True):
        self.log_root = log_root
        self.port = port
        self.auto_launch = auto_launch
        self.started = False
        self.config_copied = False
        self.run_log_dir = None
        self.project_name = None
        self.run_name = None
        self.enabled = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.started:
            return

        # 检查是否启用 TensorBoard
        report_to = args.report_to
        if 'tensorboard' not in report_to:
            self.started = True
            return

        self.enabled = True

        # from pathlib import Path
        # from datetime import datetime

        # # 1. 确定项目名和运行名，创建目录
        # self.project_name = Path(args.output_dir).name
        # self.run_name = datetime.now().strftime('%m%d-%H%M')
        # self.run_log_dir = Path(self.log_root) / self.project_name / self.run_name
        # # self.run_log_dir.mkdir(parents=True, exist_ok=True)

        # # 2. 更新 args.logging_dir，让 transformers 直接写 tb_logs
        # args.logging_dir = self.run_log_dir.as_posix()


        run_name = args.run_name
        project_name = Path(args.output_dir).name
        self.run_log_dir = Path(self.log_root) / project_name / run_name

        # 3. 启动 TensorBoard（可选）
        if not self.auto_launch: return
        try:
            import subprocess
            import webbrowser
            subprocess.Popen(
                ["tensorboard", f"--logdir={self.log_root}", f"--port={self.port}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            webbrowser.open(f"http://localhost:{self.port}")
            print(f"✅ TensorBoard 已自动启动，访问: http://localhost:{self.port}")
            print(f"   项目: {project_name}, 运行: {run_name}")
            print(f"   日志目录: {self.run_log_dir}")
        except Exception as e:
            print(f"⚠️  自动启动 TensorBoard 失败: {e}")
            print(f"   请手动执行: tensorboard --logdir={self.log_root} --port={self.port}")
            print(f"   并访问: http://localhost:{self.port}")

        self.started = True


# # alway get a copy of it 
# def get_TrainingArguments_common_params():
#     return MyDefaultTrainingArguments()

def get_Trainer_common_params(model: nn.Module, args: TrainingArguments, ds_setting: DatasetSetting):
    do_convert = next(layer.out_features for layer in reversed(list(model.modules())) if isinstance(layer, nn.Linear)) != 1
    return dict(
        model = model,
        args = args,
        train_dataset = ds_setting.train_dataset,
        eval_dataset = ds_setting.eval_dataset,
        compute_metrics = partial(compute_metrics, do_convert=do_convert),
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=3),
            TestTqdmCallback(),
            TensorBoardLauncherCallback()
        ],
    )

def trainer_init(trainer_params: dict, ds_setting: DatasetSetting):
    trainer = Trainer(**trainer_params)
    trainer.add_callback(EvalSlidingCallback(trainer, ds_setting))
    return trainer
    

def trainer_start(trainer: Trainer, ds_setting: DatasetSetting):
    trainer.evaluate()
    trainer.train()
    trainer.log_metrics('test', trainer.evaluate(ds_setting.test_dataset, metric_key_prefix='test'))
    if wandb.run: wandb.run.finish()
