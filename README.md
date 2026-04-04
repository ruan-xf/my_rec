


环境准备  
```
uv sync
```

cuda 12.7  
```
uv sync  --extra cu126
```

由于 full.parquet 过大未上传github（438MB，超过100MB都不能上传了所以还加了检查），需要先运行 data_util.py 的 save_full 函数  


evaluate的roc_auc需要加载，要设置.env（见.env-example）  


```
2/
├── common.py          # 放共用的：feature_encoders, collate_fn, 等
├── modeling_albert.py # Albert模型
├── modeling_mlp.py    # MLP模型
├── modeling_cnn.py    # CNN模型
├── train_albert.py    # Albert训练
├── train_mlp.py       # MLP训练
├── train_cnn.py       # CNN训练
```


mlp 历史平均向量  
cnn 全局最大池化 不管序列有多长，都能得到固定长度的向量，以作为其后MLP的固定维度输入  
将feature向量化的模块能够被所有模型复用  

训练看板使用tensorboard  
设置另外的logging_dir，与保存的checkpoint独立，这样一是checkpoint可被覆盖而记录仍可查阅，二是便于比较（同比 / 环比）  

目前效果  
- mlp 0.55
- cnn 0.65


- 使用新数据
- cnn kernel=3, 6
- albert


-  

负样本采样有大问题  
采样范围：1 到 5,163,070  
词表范围：0 到 3,652,290（词表ID从0开始）  

我基于训练集训练的tokenizer词表大小为3,652,291  
但采样范围却仍然使用数据集的item范围，当采样到 item_id > 3,652,290 时，tokenizer会将其映射为 <unk> (ID=1)，这意味着大约 29% 的负样本 (5163070-3652290)/5163070 ≈ 0.29 会变成 <unk> token，造成了实质上的数据泄露  


问题	影响  
29%负样本是<unk>	模型学到错误模式：正样本是真实item ID，负样本大量是1（<unk>）  
数据泄漏	模型可能学会了"item_id != 1"就预测为正，而不是真正学习用户偏好  

