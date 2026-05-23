
import matplotlib.pyplot as plt
import pickle

# 可视化不同样本的注意力矩阵
# 行：不同的样本
# 列：不同的注意力头

# 加载样本数据
with open('samples.pkl', 'rb') as f:
    samples = pickle.load(f)  # list of [num_heads, seq_len, seq_len]

num_samples = len(samples)
num_heads = samples[0].shape[0]  # 8个头

# print(f"样本数: {num_samples}")
# print(f"注意力头数: {num_heads}")
# print(f"序列长度: {samples[0].shape[1]}")

# 创建 num_samples * num_heads 的图
fig, axes = plt.subplots(num_samples, num_heads, figsize=(num_heads*2, num_samples*2))

for i in range(num_samples):
    attention_weights = samples[i]  # [num_heads, seq_len, seq_len]

    for h in range(num_heads):
        ax = axes[i, h]
        attn_matrix = attention_weights[h]  # 获取第h个头的注意力权重
        ax.imshow(attn_matrix, cmap='gray_r', aspect='auto')

        if i == 0:
            ax.set_title(f'Head {h+1}', fontsize=10)

        if h == 0:
            ax.set_ylabel(f'Sample {i+1}', rotation=0, labelpad=40, va='center', fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
# plt.savefig('attention_samples.png', dpi=150, bbox_inches='tight')
plt.show()

# print("\n可视化已保存到 attention_samples.png")
