

from transformers import PreTrainedModel

import torch
from torch import nn
from torchvision.ops import MLP

import common

class MLPModel(PreTrainedModel, common.FeatureEmbeddingMixin):
    config_class = common.MLPConfig

    def __init__(self, config: common.MLPConfig):
        super().__init__(config)
        self.all_tied_weights_keys = {}  # transformers 5.5+ 要求

        self._init_feature_embeddings(config)

        activation_layer = torch.nn.Tanh if config.activation_layer == 'tanh' else torch.nn.ReLU

        self.mlp = MLP(
            config.embedding_size * 2,  # 历史平均 + 目标特征
            config.hidden_channels + [1],
            nn.BatchNorm1d,
            activation_layer=activation_layer,
            dropout=config.dropout
        )

    def forward(self, behavior_type, category_id, item_id, labels=None, item_seq=None, **kwargs):
        concated = self._concat_feature_embeddings(behavior_type, category_id, item_id)

        # 方法：历史平均 + 目标特征
        # 由于是固定格式的数据，最后一个位置就是目标
        avg_hist = concated[:, :-1, :].mean(dim=-2)
        target_emb = concated[:, -1, :]

        # 拼接历史平均和目标特征
        combined = torch.cat([avg_hist, target_emb], dim=-1)

        logits = self.mlp(combined).squeeze(-1)

        if labels is not None:
            loss = nn.MSELoss()(logits, labels.float())
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


def model_init():
    return MLPModel(common.MLPConfig())

