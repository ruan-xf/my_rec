"""
注意力模式分析脚本
用于分析 ALBERT 模型在推荐任务中关注的模式：
1. 局部依赖（对角线附近）vs 长距离依赖
2. 同一时间步内不同特征之间的交互
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import config
import common
import utils
from modeling_albert import AlbertRec, RecConfig


def load_model(checkpoint_path: str):
    """加载训练好的模型"""
    model = AlbertRec.from_pretrained(checkpoint_path)
    model.config._attn_implementation = "eager"
    model.eval()
    return model


def get_attention_weights_via_hooks(model, inputs):
    """使用 forward hooks 获取注意力权重"""
    attention_weights = []
    handles = []

    encoder = model.albert_classifier.albert.encoder
    num_layers = model.config.num_hidden_layers

    for layer_idx in range(num_layers):
        group_idx = int(layer_idx * model.config.num_hidden_groups / model.config.num_hidden_layers)
        attention_module = encoder.albert_layer_groups[group_idx].albert_layers[0].attention

        storage = []
        attention_weights.append(storage)

        def create_hook(s):
            def hook(_module, _input, output):
                s.append(output[1].clone())
            return hook

        handle = attention_module.register_forward_hook(create_hook(storage))
        handles.append(handle)

    try:
        with torch.no_grad():
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
        for handle in handles:
            handle.remove()

    final_weights = []
    for layer_storage in attention_weights:
        if layer_storage and len(layer_storage) > 0:
            final_weights.append(layer_storage[-1])
        else:
            final_weights.append(None)

    return final_weights


def analyze_attention_pattern(attention_weights, seq_len: int):
    """
    分析注意力模式

    Returns:
        local_ratio: 局部注意力（距离<=3）的比例
        medium_ratio: 中距离注意力（3<距离<=20）的比例
        long_ratio: 长距离注意力（距离>20）的比例
    """
    # 只需要第一层（靠近输入）和最后一层（靠近输出）
    layer_indices = [0, len(attention_weights) // 2, len(attention_weights) - 1]

    results = {}

    for layer_idx in layer_indices:
        attn = attention_weights[layer_idx][0]  # (heads, seq, seq)
        num_heads = attn.shape[0]

        local_ratios = []
        medium_ratios = []
        long_ratios = []

        for h in range(num_heads):
            attn_h = attn[h].cpu().numpy()

            # 计算注意力分布
            seq_len_actual = attn_h.shape[0]

            # 排除对角线（自己关注自己）和mask的位置
            for i in range(seq_len_actual):
                row = attn_h[i]
                # 计算到其他位置的距离
                distances = np.abs(np.arange(seq_len_actual) - i)

                local_mask = distances <= 3
                medium_mask = (distances > 3) & (distances <= 20)
                long_mask = distances > 20

                # 归一化的注意力权重
                total = row.sum()
                if total > 0:
                    local_ratios.append(row[local_mask].sum() / total)
                    medium_ratios.append(row[medium_mask].sum() / total)
                    long_ratios.append(row[long_mask].sum() / total)

        results[f'layer_{layer_idx}'] = {
            'local': np.mean(local_ratios),
            'medium': np.mean(medium_ratios),
            'long': np.mean(long_ratios),
        }

        print(f"\n=== Layer {layer_idx} ===")
        print(f"  局部 (距离≤3):   {results[f'layer_{layer_idx}']['local']:.1%}")
        print(f"  中距离 (4-20):   {results[f'layer_{layer_idx}']['medium']:.1%}")
        print(f"  长距离 (>20):    {results[f'layer_{layer_idx}']['long']:.1%}")

    return results


def analyze_within_step_attention(attention_weights, seq_len: int):
    """
    分析同一时间步内不同特征之间的注意力
    每个时间步有3个特征: behavior_type, category_id, item_id

    在 attention matrix 中:
    - 位置 0, 3, 6, ... 对应 behavior_type
    - 位置 1, 4, 7, ... 对应 category_id  (如果是按特征顺序拼接)
    - 位置 2, 5, 8, ... 对应 item_id

    但由于你是把3个特征在embedding维度拼接，然后在序列维度排列，
    所以 attention matrix 的索引对应的是"时间步"，每个时间步内部没有分开
    """
    # 你的输入结构是:
    # [behavior_type_emb(20), category_id_emb(20), item_id_emb(20)] 在 embedding 维度拼接
    # 然后 sequence 维度是时间步

    # 所以 attention matrix 的每个 (i, j) 表示"时间步 i 关注时间步 j"
    # 而不是一个时间步内的3个特征之间的关注

    print("\n=== 特征维度分析 ===")
    print("你的输入是: seq_len 个时间步，每个时间步包含 3 个特征的 embedding 拼接")
    print("attention matrix 是 (seq_len, seq_len)，关注的是时间步之间的依赖")
    print("要分析特征之间的交互，需要修改输入表示方式")


def visualize_attention_distance(attention_weights, seq_len: int, save_path: str = None):
    """可视化注意力随距离的衰减"""
    # 选取第一层和最后一层
    layer_indices = [0, len(attention_weights) - 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, layer_idx in enumerate(layer_indices):
        attn = attention_weights[layer_idx][0]  # (heads, seq, seq)
        num_heads = attn.shape[0]

        # 计算平均注意力随距离的分布
        max_dist = seq_len
        distance_attention = np.zeros(max_dist)

        for h in range(num_heads):
            attn_h = attn[h].cpu().numpy()
            for i in range(seq_len):
                for j in range(seq_len):
                    dist = abs(i - j)
                    distance_attention[dist] += attn_h[i, j]

        # 归一化（每个距离的计数）
        distance_attention = distance_attention / (distance_attention.sum() + 1e-8)

        ax = axes[idx]
        ax.bar(range(max_dist), distance_attention, alpha=0.7)
        ax.set_xlabel('Distance (|i - j|)')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'Layer {layer_idx}: Attention vs Distance')
        ax.axvline(x=3, color='r', linestyle='--', label='local=3')
        ax.axvline(x=20, color='orange', linestyle='--', label='medium=20')
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_attention_heatmap(attention_weights, seq_len: int, save_path: str = None):
    """可视化原始注意力热力图"""
    layer_indices = [0, len(attention_weights) // 2, len(attention_weights) - 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, layer_idx in enumerate(layer_indices):
        attn = attention_weights[layer_idx][0, 0].cpu().numpy()  # 第一个head

        ax = axes[idx]
        im = ax.imshow(attn, cmap='viridis', aspect='auto')
        ax.set_xlabel('Position j')
        ax.set_ylabel('Position i')
        ax.set_title(f'Layer {layer_idx}: Attention (Head 1)')

        # 标注对角线（局部）区域
        ax.axhline(y=seq_len//2, color='white', linestyle='--', alpha=0.5)
        ax.axvline(x=seq_len//2, color='white', linestyle='--', alpha=0.5)

    plt.colorbar(im, ax=axes, shrink=0.6, label='Attention Weight')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def get_sample_from_test_dataset(seq_len: int = None):
    """从测试集获取样本"""
    from transformers import Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig

    class DemoModel(PreTrainedModel):
        def forward(self, behavior_type, category_id, item_id, attention_mask, token_type_ids, labels, item_seq=None, **kwargs):
            pass

    demo_model = DemoModel(PretrainedConfig())

    ds_setting = utils.DatasetSetting(900, use_sample=True)

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
        batch = next(iter(dataloader))
    else:
        for batch in dataloader:
            actual_len = (batch['attention_mask'] == 1).sum().item()
            if actual_len == seq_len:
                break
        else:
            print(f"未找到长度 {seq_len} 的样本，使用第一个样本")
            batch = next(iter(dataloader))

    return batch


if __name__ == "__main__":
    # 配置
    checkpoint_path = "checkpoints/albert_rec/checkpoint-100"
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)

    # 1. 加载模型
    print("加载模型...")
    model = load_model(checkpoint_path)
    print(f"模型层数: {model.config.num_hidden_layers}")
    print(f"注意力头数: {model.config.num_attention_heads}")

    # 2. 获取中位数长度的样本
    batch = get_sample_from_test_dataset(74)
    seq_len = (batch['attention_mask'] == 1).sum().item()
    print(f"样本序列长度: {seq_len}")

    # 3. 获取注意力权重
    print("\n获取注意力权重...")
    attention_weights = get_attention_weights_via_hooks(model, batch)

    # 4. 分析注意力模式
    print("\n" + "="*60)
    print("分析1: 局部 vs 长距离注意力")
    print("="*60)
    analyze_attention_pattern(attention_weights, seq_len)

    # 5. 分析特征维度
    analyze_within_step_attention(attention_weights, seq_len)

    # 6. 可视化
    print("\n生成可视化...")
    visualize_attention_heatmap(
        attention_weights,
        seq_len,
        # save_path=str(output_dir / "attention_heatmap.png")
    )

    visualize_attention_distance(
        attention_weights,
        seq_len,
        # save_path=str(output_dir / "attention_distance.png")
    )

    print("\n完成!")
