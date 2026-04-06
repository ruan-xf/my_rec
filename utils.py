

from functools import partial
from pathlib import Path
import evaluate
from scipy.special import softmax

import torch
from torch import nn

import transformers
from transformers import (
    Trainer, TrainingArguments,
    EarlyStoppingCallback, TrainerCallback,
    # logging,
)

# logging.enable_progress_bar()  
# logging.set_verbosity_info()

import subprocess
# import webbrowser
from tqdm.auto import tqdm
# import wandb

# from data_util import DatasetSetting


# 加载预处理的数据确实能加快训练
# 但目前只有主观体验，希望之后能够从profiler直观看出这点
from datasets import Dataset, IterableDataset
class DatasetSetting:
    """使用parquet文件的DatasetSetting"""
    def __init__(self, per_eval_size, *args, **kwargs):
        self.train_dataset = self.load_data('train')
        self._eval_dataset = self.load_data('eval').repeat(None)
        self.test_dataset = self.load_data('test')
        self.eval_iter = None
        self.per_eval_size = per_eval_size

    def load_data(self, split: str):
        return IterableDataset.from_parquet(
            f'data/processed/hf_saved/{split}.parquet',
        )

    @property
    def eval_dataset(self):
        if self.eval_iter is None:
            self.reset_eval_iter()
        return Dataset.from_dict(next(self.eval_iter))

    def reset_eval_iter(self):
        self.eval_iter = self._eval_dataset.iter(self.per_eval_size)


roc_auc_score = evaluate.load("roc_auc")
mse_metric = evaluate.load("mse")
mae_metric = evaluate.load("mae")


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
def compute_metrics(eval_pred):
    """
    新的指标计算函数：使用MSE和MAE
    适用于[0,1]连续标签的采样策略

    Args:
        eval_pred: (predictions, labels) 元组
                   predictions 已经是经过sigmoid后的[0,1]概率值
    """
    predictions, labels = eval_pred

    mse_result = mse_metric.compute(predictions=predictions, references=labels)
    mae_result = mae_metric.compute(predictions=predictions, references=labels)

    return {
        'mse': mse_result['mse'],
        'mae': mae_result['mae'],
    }

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

        # 在此处进行 project_name, run_name 的设置不会生效，所以提前设置了
        run_name = args.run_name
        project_name = Path(args.output_dir).name
        self.run_log_dir = Path(self.log_root) / project_name / run_name

        # 3. 启动 TensorBoard（可选）
        if not self.auto_launch: return
        try:
            subprocess.Popen(
                ["tensorboard", f"--logdir={self.log_root}", f"--port={self.port}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # webbrowser.open(f"http://localhost:{self.port}")
            print(f"✅ TensorBoard 已自动启动，访问: http://localhost:{self.port}")
            print(f"   项目: {project_name}, 运行: {run_name}")
            print(f"   日志目录: {self.run_log_dir}")
        except Exception as e:
            print(f"⚠️  自动启动 TensorBoard 失败: {e}")
            print(f"   请手动执行: tensorboard --logdir={self.log_root} --port={self.port}")
            print(f"   并访问: http://localhost:{self.port}")

        self.started = True

def get_Trainer_common_params(model: nn.Module, args: TrainingArguments, ds_setting: DatasetSetting):
    return dict(
        model = model,
        args = args,
        train_dataset = ds_setting.train_dataset,
        eval_dataset = ds_setting.eval_dataset,
        compute_metrics = compute_metrics,
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
    # if wandb.run: wandb.run.finish()
