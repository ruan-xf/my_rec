
import transformers
import config

from transformers.models.albert.modeling_albert import (
    AlbertForSequenceClassification,
    AlbertConfig,
)

from common import (
    seq_features,
    FeatureEmbeddingMixin,
)


class RecConfig(AlbertConfig):
    def __init__(
        self,
        num_attention_heads=12,
        embedding_size=60,
        num_labels=2,
        dropout=0.2,
        feature_vocab_sizes=config.feature_vocab_sizes,
        **kwargs
    ):
        assert embedding_size % len(seq_features) == 0
        # 1. hidden_size 必须得是 num_attention_heads 的整数倍
        # 2. ALBERT 的 FFN 中间层维度通常是 hidden_size * 4
        # 也与预训练权重的设置进行了对比
        hidden_size = num_attention_heads * 64
        intermediate_size = hidden_size*4
        super().__init__(
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            classifier_dropout_prob=dropout,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            num_labels=num_labels,
            intermediate_size=intermediate_size,
            **kwargs
        )
        self.dropout = dropout
        self.feature_vocab_sizes = feature_vocab_sizes


class AlbertRec(transformers.PreTrainedModel, FeatureEmbeddingMixin):
    config_class = RecConfig

    def __init__(self, config: RecConfig):
        super().__init__(config)

        self._init_feature_embeddings(config)

        config.vocab_size = 1
        self.albert_classifier = AlbertForSequenceClassification(config)

    def forward(self, behavior_type, category_id, item_id, attention_mask, token_type_ids, labels, item_seq=None, **kwargs):
        concated = self._concat_feature_embeddings(behavior_type, category_id, item_id)

        inputs_embeds = self.albert_classifier.albert.embeddings(inputs_embeds=concated)

        return self.albert_classifier.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )


def model_init():
    return AlbertRec(RecConfig())

