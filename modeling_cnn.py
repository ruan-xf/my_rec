

from transformers import PreTrainedModel

import torch
from torch import nn
from torchvision.ops import MLP

import common

class CNNConfig(common.MLPConfig):
    model_type = "cnn"
    kernel_size: int = 3
    out_channels: int = 64


class CNNModel(PreTrainedModel, common.FeatureEmbeddingMixin):
    config_class = CNNConfig

    def __init__(self, config: CNNConfig):
        super().__init__(config)
        self.all_tied_weights_keys = {}  # transformers 5.5+ 要求

        self._init_feature_embeddings(config)

        self.conv = nn.Conv1d(
            in_channels=config.embedding_size,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2
        )

        activation_layer = torch.nn.Tanh if config.activation_layer == 'tanh' else torch.nn.ReLU

        self.mlp = MLP(
            config.out_channels,
            config.hidden_channels + [1],
            nn.BatchNorm1d,
            activation_layer=activation_layer,
            dropout=config.dropout
        )

    def forward(self, behavior_type, category_id, item_id, labels=None, item_seq=None, **kwargs):
        concated = self._concat_feature_embeddings(behavior_type, category_id, item_id)

        # CNN需要 (batch_size, in_channels, seq_len)
        conv_input = concated.transpose(1, 2)
        conv_output = self.conv(conv_input)

        # 全局最大池化
        pooled = conv_output.max(dim=-1)[0]

        logits = self.mlp(pooled).squeeze(-1)

        if labels is not None:
            loss = nn.MSELoss()(logits, labels.float())
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


def model_init():
    return CNNModel(CNNConfig())

