import torch
import transformers
from transformers.models.albert.modeling_albert import (
    AlbertForSequenceClassification,
    AlbertConfig,
)

import common


# python MRO，AlbertConfig的同名属性会被"遮蔽"（shadowed）
# 注意 PretrainedConfig.__post_init__ 中会覆盖 num_labels
# 而且 problem_type = regression 时 num_labels = 1 不合理？也会覆盖
class RecConfig(common.ModelConfig, AlbertConfig):
    num_attention_heads: int =8
    num_labels: int =1  # 使用连续标签，设置为1
    problem_type: str = 'regression'
    # 1. hidden_size 必须得是 num_attention_heads 的整数倍
    # 2. ALBERT 的 FFN 中间层维度通常是 hidden_size * 4
    # 也与预训练权重的设置进行了对比
    hidden_size: int = num_attention_heads * 64
    intermediate_size: int = hidden_size*4
    hidden_dropout_prob: int | float = -1
    attention_probs_dropout_prob: int | float = -1
    classifier_dropout_prob: int | float = -1
    embedding_weight_decay: float = 0.0  # embedding L2 正则系数

    def __post_init__(self, **kwargs):
        super().__post_init__()
        self.num_labels = self.__class__.num_labels
        # 只有当值未设置时才使用 dropout 的默认值
        if self.hidden_dropout_prob == -1:
            self.hidden_dropout_prob = self.dropout
        if self.attention_probs_dropout_prob == -1:
            self.attention_probs_dropout_prob = self.dropout
        if self.classifier_dropout_prob == -1:
            self.classifier_dropout_prob = self.dropout

class AlbertRec(transformers.PreTrainedModel, common.FeatureEmbeddingMixin):
    config_class = RecConfig

    def __init__(self, config):
        super().__init__(config)
        self.config: RecConfig
        self.all_tied_weights_keys = {}  # transformers 5.5+ 要求

        self._init_feature_embeddings(config)

        config.vocab_size = 1
        self.albert_classifier = AlbertForSequenceClassification(config)

    def forward(self, behavior_type, category_id, item_id, attention_mask, token_type_ids, labels, item_seq=None, **kwargs):
        concated = self._concat_feature_embeddings(behavior_type, category_id, item_id)

        # 不该手动调用
        # 重复计算：两次 embeddings 调用会导致：
            # token_type embeddings 被加了两次
            # position embeddings 被加了两次
            # LayerNorm 被应用了两次
            # Dropout 被应用了两次（这会改变随机性！）
        # 语义错误：最终得到的 embedding 并不是你想要的，因为：
            # position embedding 被加了两次 → 位置信息被双重加权
            # Dropout 被应用两次 → 正则化效果过强
        # 潜在的训练不稳定：由于 Dropout 的随机性，两次应用会导致梯度计算不一致
        # inputs_embeds = self.albert_classifier.albert.embeddings(inputs_embeds=concated)

        out = self.albert_classifier.forward(
            inputs_embeds=concated,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        out.logits = out.logits.squeeze(-1)

        # 添加 embedding L2 正则
        if self.config.embedding_weight_decay > 0 and out.loss is not None:
            l2_reg = sum(
                torch.norm(embedding.weight, p=2)
                for embedding in self.feature_embeddings
            )
            out.loss = out.loss + self.config.embedding_weight_decay * l2_reg

        return out


def model_init():
    return AlbertRec(RecConfig())


