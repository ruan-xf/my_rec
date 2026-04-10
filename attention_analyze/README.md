# Attention 分析说明

## 概述

本目录用于存储和分析 ALBERT 模型在推荐任务中的注意力机制。  

## 关键文件

| 文件 | 说明 |
|-----|------|
| `analyze_attention.py` | 官方分析脚本 |
| `visualize_attention.py` | 可视化脚本 |

## ALBERT 注意力机制

### 1. 模型结构与参数共享机制

ALBERT 的核心特性是**参数共享**，通过**重复使用层对象**实现。

#### 基本概念
- **内存对象**：在内存中创建的独立层对象，每个都有自己的参数
- **前向传递**：模型推理时的计算步骤，会调用层对象

#### 配置示例（简化情况）
假设 `inner_group_num=1`（每个组只有1个层）：
```
num_hidden_layers = 12  # 前向传递次数
num_hidden_groups = 3   # 层组数
inner_group_num = 1     # 每组内层数

内存中：只有 3 个独立的 AlbertLayerGroup 对象
前向传递时：
  layer 0-3  →  重复使用 group[0]（4次前向传递调用同一对象）
  layer 4-7  →  重复使用 group[1]（4次前向传递调用同一对象）
  layer 8-11 →  重复使用 group[2]（4次前向传递调用同一对象）
```

#### 关键理解
- **参数共享** = 同一个内存对象被多次前向传递调用
- **不是**多个内存对象共享同一块参数内存
- 每个内存对象（AlbertLayer）有自己独立的参数存储

### 2. 注意力权重捕获

Hook 捕获位置：`AlbertAttention.forward` 的返回值  

```python
# transformers/models/albert/modeling_albert.py:188-204
attn_output, attn_weights = attention_interface(...)  # hook 捕获这里
return attn_output, attn_weights
```

其中 `attn_weights` 是 `eager_attention_forward` 中 softmax 后的结果：  
```python
# modeling_albert.py:133
attn_weights = nn.functional.softmax(attn_weights, dim=-1)
return attn_output, attn_weights
```

#### 完整数据流

```
输入 embedding
    ↓
create_bidirectional_mask()  ← 注意：这里只创建 mask，不产生注意力权重
    ↓
AlbertAttention.forward (hook 挂载点)
    ├── query = linear(hidden_states)    # Q 权重
    ├── key   = linear(hidden_states)    # K 权重
    ├── value = linear(hidden_states)    # V 权重
    │
    └── eager_attention_forward
        ├── attn_scores = Q @ K^T / sqrt(d)     # 原始注意力分数
        ├── attn_weights = softmax(attn_scores) # softmax 后的权重 ← hook 捕获这里
        └── attn_output = attn_weights @ V      # 加权求和
    ↓
返回 (attn_output, attn_weights)
    ↓
AlbertLayer.forward
    └── ffn(attn_output)  ← attn_weights 被丢弃，不往下传
    ↓
后续层...
```

#### 计算图
前向  
```
Q, K, V (可学习参数)
   ↓
attn_weights = softmax(Q@K^T)  ← 依赖 Q, K
   ↓
attn_output = attn_weights @ V ← 依赖 V 和 attn_weights
   ↓
后续网络
```

反向  
```
loss
    ↓
attn_output = attn_weights @ V
    ├── ∂loss/∂attn_output → 传给后续网络
    ├── ∂loss/∂V ← 梯度流回 V 权重
    └── ∂loss/∂attn_weights ← 梯度通过链式法则计算

attn_scores = Q @ K^T / sqrt(d)
attn_weights = softmax(attn_scores)
    └── 梯度通过链式法则流回 Q、K 权重

最终梯度流回：query.weight, key.weight, value.weight, dense.weight
```

**注意**：反向传播时，`attn_weights` 的梯度会通过 `attn_output` 和 `attn_scores` 计算，但不需要显式保存。梯度直接通过 `attn_output` 流向可学习参数。

### 3. attn_output vs attn_weights

| | attn_weights | attn_output |
|--|-------------|-------------|
| 含义 | 每个位置关注其他位置的程度 | 加权后的 context 向量 |
| Shape | `(batch, heads, seq, seq)` | `(batch, seq, hidden_size)` |
| 用途 | 分析模型行为 | 传递给后续网络 |
| 梯度 | 计算但不保存（中间变量） | 保存并用于反向传播 |

**关于 attn_weights 在反向传播中的作用**：  

- `attn_weights` 是 Q、K、V 的函数输出（中间变量），不是可学习参数
- 反向传播时梯度通过 `attn_output` 计算，然后通过 `attn_weights` 的链式法则流向 Q、K 权重
- `attn_weights` 本身是输入相关的动态值，不是静态的模型权重
- 在分析时捕获的 `attn_weights` 是前向传播的结果，用于了解模型在特定输入上的关注模式

## 代码中的分析策略

### 关键理解
- ALBERT 的注意力共享是**跨前向传递**的共享，不是**内存对象**的共享
- 每个 `AlbertLayer` 对象在内存中是独立的，有自己的注意力参数
- 多个前向传递可能使用同一个 `AlbertLayer` 对象（重复使用）

### 应该分析哪些注意力权重？

假设配置：`num_hidden_layers=12, num_hidden_groups=3, inner_group_num=2`

**内存中的独立对象**：
```python
# 总共有 3 groups × 2 inner layers = 6 个独立的 AlbertLayer
# 每个 AlbertLayer 有自己的 AlbertAttention
total_unique_layers = num_hidden_groups * inner_group_num  # 3*2=6
```

**前向传递的安排**：
```python
# 前向传播时，每个 layer 索引 i 对应：
layer_in_group = i % (num_hidden_layers // num_hidden_groups)  # 在组内的位置
group_idx = int(i / (num_hidden_layers / num_hidden_groups))    # 属于哪个组

# 然后使用对应组内的特定层
inner_layer_idx = layer_in_group % inner_group_num
layer_to_use = albert_layer_groups[group_idx].albert_layers[inner_layer_idx]
```

### 分析策略

**如果要分析所有可能的注意力模式**：
```python
# 分析每个唯一的 AlbertLayer
for group_idx in range(num_hidden_groups):
    for inner_idx in range(inner_group_num):
        attention_module = encoder.albert_layer_groups[group_idx].albert_layers[inner_idx].attention
        # 挂载 hook 或分析这个唯一的注意力模块
```

**如果只关心前向传递中的不同层**：
```python
# 分析前向传播中实际使用的不同层
# 需要在前向传播过程中根据 layer_idx 动态判断
used_layers = set()
for layer_idx in range(num_hidden_layers):
    group_idx = int(layer_idx / (num_hidden_layers / num_hidden_groups))
    inner_idx = (layer_idx % (num_hidden_layers // num_hidden_groups)) % inner_group_num
    used_layers.add((group_idx, inner_idx))

# used_layers 包含所有在实际前向中使用的层组合
```

### 分析/可视化时的层选择

**取决于分析目的**：

1. **分析所有独立的内存对象（最全面）**：
```python
layer_indices = list(range(num_hidden_groups * inner_group_num))
# 即 [0,1,2,3,4,5] 对应6个独立的层对象
```

2. **分析简化（如果 inner_group_num=1）**：
```python
layer_indices = list(range(num_hidden_groups))
# 即 [0,1,2] 对应3个不同的组（每个组只有1个层）
```

3. **分析前向传递的代表性点**：
```python
# 选择前向传播中的关键层索引
important_layer_indices = [0, num_hidden_layers//2, num_hidden_layers-1]
# 即 [0,6,11]
# 但需要映射到对应的底层对象：
for layer_idx in important_layer_indices:
    group_idx = int(layer_idx / (num_hidden_layers / num_hidden_groups))
    inner_idx = (layer_idx % (num_hidden_layers // num_hidden_groups)) % inner_group_num
    unique_idx = group_idx * inner_group_num + inner_idx
    # unique_idx 是内存中独立对象的索引
```

**建议**：对于 ALBERT 注意力分析，推荐方法1，因为它覆盖了所有可能的注意力模式。

## 分析维度

### 1. 注意力熵分析

衡量注意力分布的"聚焦程度"：  

```python
for h in range(num_heads):
    attn_h = attn[h].cpu().numpy()
    for i in range(seq_len):
        row = attn_h[i]
        row = row / (row.sum() + 1e-8)  # 归一化为概率分布
        entropy = -np.sum(row * np.log(row + 1e-8))  # 香农熵
        entropies.append(entropy)

print(f'平均熵: {np.mean(entropies):.3f} (最大可能: {np.log(seq_len):.3f})')
print(f'相当于均匀分布在 ~{np.exp(np.mean(entropies)):.1f} 个位置上')
```

**量化意义**：  

| 熵值 | 含义 | 推荐场景解读 |
|-----|------|-------------|
| 熵 ≈ 0 | 完全聚焦 1 个位置 | 强局部依赖，可能过拟合 |
| 熵适中 | 适度关注多个位置 | 正常，关注关键 item |
| 熵 ≈ log(seq_len) | 均匀分布 | 注意力分散，可能欠拟合 |

- 熵 = 0：只关注 1 个位置
- 熵 = log(20) ≈ 3：均匀分布在 20 个位置
- 你的模型熵值反映注意力有多"集中"或"分散"

### 2. 距离分析

按位置距离分类统计注意力权重：  

```python
local_ratios = []
medium_ratios = []
long_ratios = []

for h in range(num_heads):
    attn_h = attn[h].cpu().numpy()
    for i in range(seq_len):
        for j in range(seq_len):
            dist = abs(i - j)
            weight = attn_h[i, j]
            if dist <= 3:
                local_ratios.append(weight)
            elif dist <= 20:
                medium_ratios.append(weight)
            else:
                long_ratios.append(weight)

total = sum(local_ratios) + sum(medium_ratios) + sum(long_ratios)
print(f'局部 (距离≤3):   {sum(local_ratios)/total:.1%}')
print(f'中距离 (4-20):   {sum(medium_ratios)/total:.1%}')
print(f'长距离 (>20):    {sum(long_ratios)/total:.1%}')
```

**量化意义**：  

| 距离范围 | 推荐场景解读 |
|---------|-------------|
| 局部 (≤3) | 捕获 item 内的短时依赖、相邻行为 |
| 中距离 (4-20) | 中等跨度的行为模式 |
| 长距离 (>20) | 跨 item 的协同模式、长期兴趣 |

- 高局部占比 = 强时序局部性
- 高长距离占比 = 建模远距离依赖的能力

### 3. 对角线 vs 非对角线

另一种衡量"局部性"的方式：  

```python
diag_sum = 0
off_diag_sum = 0

for h in range(num_heads):
    attn_h = attn[h].cpu().numpy()
    diag_sum += np.trace(attn_h)  # 对角线元素之和
    off_diag_sum += (attn_h.sum() - np.trace(attn_h))  # 非对角线之和

total = diag_sum + off_diag_sum
print(f'对角线(自身+近邻): {diag_sum/total:.1%}')
print(f'非对角线(远距离): {off_diag_sum/total:.1%}')
```

**量化意义**：  

| 类型 | 含义 |
|-----|------|
| 对角线 | 自身关注自己 + 邻近位置 |
| 非对角线 | 远距离依赖 |

- 高对角线 = 强局部性（"近邻优先"）
- 高非对角线 = 远距离依赖建模能力

## 总结

1. **理解 ALBERT 的共享机制**：
   - 跨层共享：多个前向传递使用同一个层对象（参数复用）
   - 组内独立：每个 `AlbertLayerGroup` 内有多个独立的 `AlbertLayer` 对象

2. **分析策略**：
   - 如果 `inner_group_num=1`：只需要分析 `num_hidden_groups` 个不同的注意力权重
   - 如果 `inner_group_num>1`：需要分析 `num_hidden_groups * inner_group_num` 个不同的注意力权重
   - 捕获的是 softmax 后的注意力分布（输入相关），不是模型参数

3. **关键公式**：
   ```python
   # 正确的 group 索引计算（来自 transformers 源码）
   group_idx = int(layer_idx / (num_hidden_layers / num_hidden_groups))
   
   # 组内层索引计算
   layer_in_group = layer_idx % (num_hidden_layers // num_hidden_groups)
   inner_idx = layer_in_group % inner_group_num
   ```

4. **实践建议**：
   - 先检查模型的 `inner_group_num` 配置
   - 分析所有独立的 `AlbertLayer` 对象以获得完整视图
   - 注意注意力权重是动态计算的，每次前向传播都可能不同
