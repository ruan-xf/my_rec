
import matplotlib.pyplot as plt
import torch
from pathlib import Path


# 某个 head 的完整注意力矩阵
def get_attn_plot(attn):
    plt.imshow(attn, cmap='viridis', aspect='auto')
    plt.colorbar()
    return plt.gcf()

    
seq_lens = [3,10,30,74,150]
    
# 创建 seq_len * num_head 的图
num_head = 8
fig, axes = plt.subplots(len(seq_lens), num_head, figsize=(num_head*2, len(seq_lens)*2))

for i in range(len(seq_lens)):
    logdir = Path(f'logs/seqlen={seq_lens[i]}')
    attention_weights = torch.load(logdir / 'attention_weight.pt', weights_only=False)

    for h in range(num_head):
        ax = axes[i, h]
        attn_matrix = attention_weights[h]  # 获取第h个头的注意力权重
        ax.imshow(attn_matrix, cmap='viridis', aspect='auto')

        if i == 0:
            ax.set_title(f'Head {h+1}')

        if h == 0:
            ax.set_ylabel(f'seqlen={seq_lens[i]}', rotation=0, labelpad=40, va='center', fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])  # 只隐藏刻度，保留y轴标签

plt.tight_layout()
plt.show()
