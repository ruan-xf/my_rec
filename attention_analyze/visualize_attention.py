"""
注意力权重可视化脚本
用于分析 ALBERT 模型在推荐任务中关注序列的哪些部分
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from transformers import Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig

import config
import common
import utils
from modeling_albert import AlbertRec, RecConfig


def load_model(checkpoint_path: str):
    """加载训练好的模型"""
    model = AlbertRec.from_pretrained(checkpoint_path)
    # 必须使用 eager attention 才能获取注意力权重
    model.config._attn_implementation = "eager"
    model.eval()
    return model


def get_attention_weights_via_hooks(model, inputs):
    """
    使用 forward hooks 获取注意力权重
    因为 AlbertLayer 丢弃了 attn_weights，需要从 AlbertAttention 捕获
    """
    attention_weights = []
    handles = []

    # 注册 hook 到 AlbertAttention 层
    # 结构: encoder -> albert_layer_groups[0] -> albert_layers[0,1,...]
    encoder = model.albert_classifier.albert.encoder
    num_layers = model.config.num_hidden_layers

    for layer_idx in range(num_layers):
        # 计算 group_idx
        group_idx = int(layer_idx * model.config.num_hidden_groups / model.config.num_hidden_layers)
        attention_module = encoder.albert_layer_groups[group_idx].albert_layers[0].attention

        storage = []
        attention_weights.append(storage)

        def create_hook(s):
            def hook(_module, _input, output):
                # output = (attn_output, attn_weights)
                s.append(output[1].clone())
            return hook

        handle = attention_module.register_forward_hook(create_hook(storage))
        handles.append(handle)

    try:
        with torch.no_grad():
            # 执行 forward
            concated = model._concat_feature_embeddings(
                inputs['behavior_type'],
                inputs['category_id'],
                inputs['item_id']
            )

            embedding_output = model.albert_classifier.albert.embeddings(inputs_embeds=concated)

            from transformers.models.albert.modeling_albert import create_bidirectional_mask
            attention_mask = create_bidirectional_mask(
                config=model.config,
                inputs_embeds=embedding_output,
                attention_mask=inputs['attention_mask'],
            )

            model.albert_classifier.albert.encoder(
                embedding_output,
                attention_mask,
                token_type_ids=inputs['token_type_ids'],
            )
    finally:
        # 移除 hooks
        for handle in handles:
            handle.remove()

    # 整理注意力权重
    final_weights = []
    for layer_storage in attention_weights:
        if layer_storage and len(layer_storage) > 0:
            final_weights.append(layer_storage[-1])
        else:
            final_weights.append(None)

    return final_weights


def visualize_attention(attention_weights, token_labels, save_path: str = None):
    """
    可视化注意力权重

    Args:
        attention_weights: tuple of (batch, heads, seq, seq)
        token_labels: 每个位置的 token 标签列表
        save_path: 保存路径
    """
    # 取第一个样本的第一个 head 的注意力
    # attention_weights 包含所有层的注意力
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]

    # 创建一个大图，显示所有层的注意力
    fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=(20, 8))
    axes = axes.flatten()

    for layer_idx in range(num_layers):
        # 取第一个样本，第一个 head
        attn = attention_weights[layer_idx][0, 0].cpu().numpy()

        ax = axes[layer_idx]
        im = ax.imshow(attn, cmap='viridis', aspect='auto')

        # 设置标签
        ax.set_xticks(range(len(token_labels)))
        ax.set_yticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(token_labels, fontsize=8)

        ax.set_title(f'Layer {layer_idx + 1}', fontsize=10)

    # 隐藏多余的子图
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Attention Weights Across Layers', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_single_layer(attention_weights, layer_idx: int, token_labels, save_path: str = None):
    """可视化单层的所有注意力头"""
    attn_layer = attention_weights[layer_idx]
    num_heads = attn_layer.shape[1]

    fig, axes = plt.subplots(2, num_heads // 2, figsize=(20, 8))
    axes = axes.flatten()

    for head_idx in range(num_heads):
        attn = attn_layer[0, head_idx].cpu().numpy()

        ax = axes[head_idx]
        im = ax.imshow(attn, cmap='viridis', aspect='auto')

        ax.set_title(f'Head {head_idx + 1}', fontsize=10)
        ax.axis('off')

    plt.suptitle(f'Layer {layer_idx + 1} - All Attention Heads', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def get_token_labels_from_batch(batch, feature_encoders):
    """
    从 batch 中获取每个位置的 token 标签

    返回:
        token_labels: dict，key 是特征名，value 是 token 列表
    """
    labels = {}
    for col in config.seq_features:
        # 解码 input_ids
        ids = batch[col][0]  # batch_size=1
        tokens = feature_encoders[col].decode(ids).split()
        labels[col] = tokens

    return labels

def get_sample_from_test_dataset(seq_len: int = None):
    """
    从测试集获取样本，参考 test_collate_and_model.py 的方式

    Args:
        seq_len: 如果指定，则筛选对应长度的样本

    Returns:
        sample_batch: 单个样本的 batch 字典
    """
    # 创建 DemoModel（只用于获取 batch）
    class DemoModel(PreTrainedModel):
        def forward(self, behavior_type, category_id, item_id, attention_mask, token_type_ids, labels, item_seq=None, **kwargs):
            pass

    demo_model = DemoModel(PretrainedConfig())

    # 创建 DatasetSetting
    ds_setting = utils.DatasetSetting(900, use_sample=True)

    # 创建 Trainer 获取数据
    args = TrainingArguments(
        per_device_eval_batch_size=1,
        eval_strategy='steps',
        save_strategy='no',
        report_to='none'
    )

    trainer = Trainer(
        demo_model,
        args=args,
        eval_dataset=ds_setting.test_dataset,
        data_collator=common.collate_fn,
    )

    dataloader = trainer.get_eval_dataloader()

    if seq_len is None:
        # 取第一个样本
        batch = next(iter(dataloader))
    else:
        # 遍历找到指定长度的样本
        for batch in dataloader:
            # 检查序列长度（通过 attention_mask）
            actual_len = (batch['attention_mask'] == 1).sum().item()
            if actual_len == seq_len:
                break
        else:
            print(f"未找到长度 {seq_len} 的样本，使用第一个样本")
            batch = next(iter(dataloader))

    return batch


def prepare_inputs_from_batch(batch):
    """直接使用 collate_fn 处理后的 batch"""
    # batch 已经由 collate_fn 处理好了，直接返回
    return batch

if __name__ == "__main__":
    # 配置
    checkpoint_path = "checkpoints/albert_rec/checkpoint-100"  # 你的 checkpoint 路径
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)

    # 1. 加载模型
    print("加载模型...")
    model = load_model(checkpoint_path)
    print(f"模型层数: {model.config.num_hidden_layers}")
    print(f"注意力头数: {model.config.num_attention_heads}")

    # 2. 获取中位数长度的样本
    batch = get_sample_from_test_dataset(74)

    # 3. 获取 token 标签（用于显示）
    token_labels = get_token_labels_from_batch(batch, common.feature_encoders)
    print(f"\nToken 标签示例 (item_id 前5个): {token_labels['item_id'][:5]}")
    print(f"Token 标签示例 (category_id 前5个): {token_labels['category_id'][:5]}")
    print(f"Token 标签示例 (behavior_type): {token_labels['behavior_type']}")

    # 4. 获取注意力权重
    print("\n获取注意力权重...")
    attention_weights = get_attention_weights_via_hooks(model, batch)
    print(f"注意力权重层数: {len(attention_weights)}")
    print(f"每层形状: {attention_weights[0].shape}")  # (batch, heads, seq, seq)

    # 5. 可视化
    print("\n生成可视化...")

    # 使用 category_id 作为 token 标签（更直观）
    vis_token_labels = token_labels['category_id']

    # 5.1 可视化所有层的第一个头的注意力
    visualize_attention(
        attention_weights,
        vis_token_labels,
        # save_path=str(output_dir / "all_layers_attention.png")
    )

    # 5.2 可视化第一层的所有头
    visualize_single_layer(
        attention_weights,
        layer_idx=0,
        token_labels=vis_token_labels,
        # save_path=str(output_dir / "layer1_all_heads.png")
    )

    # 5.3 可视化最后一层的所有头（通常更有语义信息）
    visualize_single_layer(
        attention_weights,
        layer_idx=len(attention_weights) - 1,
        token_labels=vis_token_labels,
        # save_path=str(output_dir / f"layer{len(attention_weights)}_all_heads.png")
    )

    print("\n完成!")
