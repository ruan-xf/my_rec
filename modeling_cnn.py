
from transformers import PretrainedConfig, PreTrainedModel

import torch
from torch import nn
from torchvision.ops import MLP
import config

from common import (
    seq_features,
    FeatureEmbeddingMixin,
)


class CNNConfig(PretrainedConfig):
    model_type = "cnn"

    def __init__(
        self, *,
        feature_vocab_sizes=config.feature_vocab_sizes,
        embedding_size: int = 60,
        hidden_channels=[128, 64, 32],
        activation_layer='relu',
        dropout=0.1,
        pad_token_id=0,
        kernel_size: int = 3,
        out_channels: int = 64,
        **kwargs,
    ):
        assert embedding_size % len(seq_features) == 0
        self.feature_vocab_sizes = feature_vocab_sizes
        self.embedding_size = embedding_size
        self.hidden_channels = hidden_channels
        self.activation_layer = activation_layer
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        super().__init__(**kwargs)


class CNNModel(PreTrainedModel, FeatureEmbeddingMixin):
    config_class = CNNConfig

    def __init__(self, config: CNNConfig):
        super().__init__(config)

        self._init_feature_embeddings(config)

        self.conv = nn.Conv1d(
            in_channels=config.embedding_size,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2
        )

        assert config.activation_layer in ['relu', 'tanh']
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
            loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            return {"loss": loss, "logits": torch.sigmoid(logits)}

        return {"logits": torch.sigmoid(logits)}


def model_init(**kwargs):
    return CNNModel(CNNConfig(**kwargs))

