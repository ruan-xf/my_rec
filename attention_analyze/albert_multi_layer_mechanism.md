# ALBERT 多层参数共享机制详解

本文档详细说明 ALBERT 在多层配置下的参数共享机制和注意力分析策略。

## 模型结构与参数共享机制

### 基本概念
- **内存对象**：在内存中创建的独立层对象，每个都有自己的参数
- **前向传递**：模型推理时的计算步骤，会调用层对象

### 多层配置示例

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

### 关键理解
- **参数共享** = 同一个内存对象被多次前向传递调用
- **不是**多个内存对象共享同一块参数内存
- 每个内存对象（AlbertLayer）有自己独立的参数存储

## 应该分析哪些注意力权重？

假设配置：`num_hidden_layers=12, num_hidden_groups=3, inner_group_num=2`

### 内存中的独立对象

```python
# 总共有 3 groups × 2 inner layers = 6 个独立的 AlbertLayer
# 每个 AlbertLayer 有自己的 AlbertAttention
total_unique_layers = num_hidden_groups * inner_group_num  # 3*2=6
```

### 前向传递的安排

```python
# 前向传播时，每个 layer 索引 i 对应：
layer_in_group = i % (num_hidden_layers // num_hidden_groups)  # 在组内的位置
group_idx = int(i / (num_hidden_layers / num_hidden_groups))    # 属于哪个组

# 然后使用对应组内的特定层
inner_layer_idx = layer_in_group % inner_group_num
layer_to_use = albert_layer_groups[group_idx].albert_layers[inner_layer_idx]
```

## 分析策略

### 策略1：分析所有独立的内存对象（最全面）

```python
# 分析每个唯一的 AlbertLayer
for group_idx in range(num_hidden_groups):
    for inner_idx in range(inner_group_num):
        attention_module = encoder.albert_layer_groups[group_idx].albert_layers[inner_idx].attention
        # 挂载 hook 或分析这个唯一的注意力模块
```

### 策略2：分析前向传递中的不同层

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

## 分析/可视化时的层选择

**取决于分析目的**：

### 1. 分析所有独立的内存对象（最全面）

```python
layer_indices = list(range(num_hidden_groups * inner_group_num))
# 即 [0,1,2,3,4,5] 对应6个独立的层对象
```

### 2. 分析简化（如果 inner_group_num=1）

```python
layer_indices = list(range(num_hidden_groups))
# 即 [0,1,2] 对应3个不同的组（每个组只有1个层）
```

### 3. 分析前向传递的代表性点

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

## 关键公式

```python
# 正确的 group 索引计算（来自 transformers 源码）
group_idx = int(layer_idx / (num_hidden_layers / num_hidden_groups))

# 组内层索引计算
layer_in_group = layer_idx % (num_hidden_layers // num_hidden_groups)
inner_idx = layer_in_group % inner_group_num
```

## 实践建议

1. 先检查模型的 `inner_group_num` 配置
2. 分析所有独立的 `AlbertLayer` 对象以获得完整视图
3. 注意注意力权重是动态计算的，每次前向传播都可能不同
