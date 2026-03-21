# 优化数据加载提升GPU利用率

## 问题
训练时GPU显存占用较高，但GPU利用率低，工作间隔过长。瓶颈在于数据加载速度跟不上GPU计算速度。

## 改动

### 1. config.py
- 新增 `item_id_range` 配置项，定义负样本采样范围 `(1, 5163070)`

### 2. data_util.py

#### 快速负样本采样 (`_fast_neg_sample`)
- 使用 numpy 批量采样替代原有的无限重试循环
- 预生成负样本池并缓存
- 最多重试10次，避免无限阻塞

#### 数据预取机制 (`PrefetchIterableDataset`)
- 使用后台线程 + `queue.Queue` 实现数据预取
- 默认预取100个样本
- 让数据生成与模型训练并行执行

#### DatasetSetting 更新
- 构造函数新增 `prefetch_size` 参数
- 所有 IterableDataset 用 PrefetchIterableDataset 包装

## 效果预期
- GPU等待数据时间大幅减少
- GPU利用率提升
- 整体训练速度加快
