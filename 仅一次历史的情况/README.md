


我不确定现在只一个元素的序列还会不会出问题  
```

# batch = diagnose_embedding_inputs()
# {'labels': tensor([0, 1], device='cuda:0'),
#  'item': tensor([ 622323, 3372877], device='cuda:0'),
#  'behavior_type': tensor([], device='cuda:0', size=(2, 0)),
#  'category_id': tensor([], device='cuda:0', size=(2, 0)),
#  'item_id': tensor([], device='cuda:0', size=(2, 0)),
#  'token_type_ids': tensor([[1.],
#          [1.]], device='cuda:0'),
#  'attention_mask': tensor([[1.],
#          [1.]], device='cuda:0')}


# df['item_seq_len'].value_counts().filter(pl.col('item_seq_len') == 1)
# item_seq_len	count
# u32	u32
# 1	92
```


测试流程  
- 手工筛选出一个
- 包装为dataset
- 按 test_collate_and_model 的流程
- 要使用真实的模型

