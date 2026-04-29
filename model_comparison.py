# from data_util import DatasetSetting
# import data_util
# df = data_util.read_full()
# df['item_seq_len'].value_counts(sort=True)




import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import data_util
import modeling_albert
import transformers
from datasets import IterableDataset

import modeling_cnn
import modeling_mlp


models = {
    'cnn': modeling_cnn.CNNModel.from_pretrained('checkpoints/cnn_rec/checkpoint-3000'),
    'mlp': modeling_mlp.MLPModel.from_pretrained('checkpoints/mlp_rec/checkpoint-2200'),
    'albert': modeling_albert.AlbertRec.from_pretrained('checkpoints/albert_rec/checkpoint-3000'),
}


import utils
# test_dataset = utils.DatasetSetting.load_data('test')
test_dataset = data_util.DatasetSetting(20).test_dataset

import common
import utils

resdir = Path('final_results')
trainer = transformers.Trainer(
    models['albert'],
    transformers.TrainingArguments(per_device_eval_batch_size=100, report_to='none'),
    data_collator=common.collate_fn,
    compute_metrics = utils.compute_metrics,
    callbacks = [
        utils.TestTqdmCallback(),
    ],
)


all_metrics = []

for name, model in models.items():
    trainer.model = model
    output = trainer.predict(test_dataset, metric_key_prefix='pred')
    torch.save(output.predictions, resdir / f'{name}_output.pt')
    all_metrics.append(output.metrics)
    trainer.log_metrics(name, output.metrics)
    
metrics_df = pd.DataFrame(all_metrics)
metrics_df.index = models

metrics_df.to_csv(resdir / 'metrics.csv')
