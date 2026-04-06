


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
