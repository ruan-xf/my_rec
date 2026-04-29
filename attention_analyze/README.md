# Attention 分析说明

## 概述

本目录用于存储和分析 ALBERT 模型在推荐任务中的注意力机制。  

## 关键文件

| 文件 | 说明 |
|-----|------|
| `analyze_attention.py` | 官方分析脚本 |
| `visualize_attention.py` | 可视化脚本 |

## ALBERT 注意力机制

### 当前模型配置

本项目使用的 ALBERT 模型是**单层配置**：  
- `num_hidden_layers = 1`（前向传递次数）
- `num_hidden_groups = 1`（层组数）
- `inner_group_num = 1`（每组内层数）

这意味着只有**1个 AlbertLayer**，参数共享机制在这个配置下不体现。对于多层配置的详细说明，请参考 [albert_multi_layer_mechanism.md](albert_multi_layer_mechanism.md)。  

### 注意力权重捕获

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

### 单层模型的分析

对于单层配置（`num_hidden_layers=1, num_hidden_groups=1, inner_group_num=1`）：  
- 只需要分析第0层的注意力权重
- 所有分析都针对这一个唯一的 AlbertLayer

### 简化的注意力获取

```python
# 直接访问唯一的注意力模块
attention_module = model.albert_classifier.albert.encoder.albert_layer_groups[0].albert_layers[0].attention

# 使用 hook 捕获注意力权重
def hook(_module, _input, output):
    attention_weight = output[1].clone()  # (batch, heads, seq, seq)

handle = attention_module.register_forward_hook(hook)
```

对于多层配置的分析策略，请参考 [albert_multi_layer_mechanism.md](albert_multi_layer_mechanism.md)。  




验证多头注意力优势需要的三个核心点  
1. 多尺度感受野并存

检测不同头的"等效感受野"（关注多少个位置）  
证明：有的头只关注1个位置，有的头关注60+个位置  

2. 动态内容选择

同一头在不同输入样本下，热点位置是否变化  
证明：位置偏置 vs 内容相关的头  

3. 直接长距离建模

计算注意力在长距离（>20步）的比例  
证明：可以直接关注远距离，不需要堆叠  

【验证1】多尺度感受野并存  
(通过注意力熵衡量关注范围)  
  头类型分布:
    Head 0: 目标导向, 等效感受野=1.0  
    Head 1: 历史关系, 等效感受野=10.5  
    Head 2: 历史关系, 等效感受野=19.7  
    Head 3: 目标导向, 等效感受野=1.0  
    Head 4: 目标导向, 等效感受野=1.0  
    Head 5: 历史关系, 等效感受野=4.9  
    Head 6: 目标导向, 等效感受野=1.0  
    Head 7: 历史关系, 等效感受野=41.2  
  最小感受野: 1.0  
  最大感受野: 41.2  
  ✅ 证明：不同头的感受野差异 41.2x，且存在目标导向与历史关系两种模式  

【验证2】动态内容选择  
  Head 0: 内容相关 (位置方差 482.6)  
  Head 1: 内容相关 (位置方差 493.0)  
  Head 2: 内容相关 (位置方差 302.6)  
  Head 3: 内容相关 (位置方差 466.1)  
  Head 4: 内容相关 (位置方差 408.6)  
  Head 5: 内容相关 (位置方差 531.1)  
  Head 6: 内容相关 (位置方差 492.0)  
  Head 7: 内容相关 (位置方差 273.1)  

【验证3】直接长距离建模  
  Head 0: 长距离注意力占比 70.3%  
  Head 1: 长距离注意力占比 51.8%  
  Head 2: 长距离注意力占比 52.0%  
  Head 3: 长距离注意力占比 8.1%  
  Head 4: 长距离注意力占比 18.6%  
  Head 5: 长距离注意力占比 43.2%  
  Head 6: 长距离注意力占比 56.4%  
  Head 7: 长距离注意力占比 53.1%  

✅ 证明：注意力机制可直接建模长距离依赖  