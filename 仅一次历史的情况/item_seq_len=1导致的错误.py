


# RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
def diagnose_embedding_inputs(max_batches=None, start_from_batch=0):
    """
    遍历 train_dataset 检查传递给 embeddings 的数据类型
    
    Args:
        max_batches: 最多检查多少个 batch，None 表示检查所有
        start_from_batch: 从第几个 batch 开始检查（用于定位问题 batch）
    """
    
    # 创建与训练时相同的设置
    args = BertTrainingArguments(report_to='none')
    args.per_device_train_batch_size = 2  # 使用与训练相同的 batch size
    ds_setting = utils.DatasetSetting(False, 48*80)
    
    # 创建 trainer 以获取 dataloader
    trainer_params = dict(
        **utils.get_Trainer_common_params(model_init(), args, ds_setting),
        data_collator=collate_fn,
    )
    trainer = utils.trainer_init(trainer_params, ds_setting)
    
    dataloader = trainer.get_train_dataloader()
    
    print("=" * 80)
    print(f"开始诊断 embedding 输入数据类型")
    print("=" * 80)
    
    batch_idx = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx < start_from_batch:
            continue
        if max_batches and (batch_idx - start_from_batch) >= max_batches:
            break
            
        # 检查每个需要传递给 embedding 的字段
        embedding_fields = ['behavior_type', 'category_id', 'item_id', 'item']
        issues = []
        
        for field in embedding_fields:
            if field not in batch:
                issues.append(f"  ✗ {field}: 缺失")
                continue
                
            tensor = batch[field]
            dtype = tensor.dtype
            shape = tensor.shape
            
            # 检查是否是 long 类型
            if dtype != torch.long:
                issues.append(f"  ✗ {field}: dtype={dtype} (应该是 torch.long), shape={shape}")
                # 检查是否有 NaN 或 Inf
                if torch.is_floating_point(tensor):
                    nan_count = torch.isnan(tensor).sum().item()
                    inf_count = torch.isinf(tensor).sum().item()
                    if nan_count > 0:
                        issues.append(f"     包含 {nan_count} 个 NaN")
                    if inf_count > 0:
                        issues.append(f"     包含 {inf_count} 个 Inf")
                    # 显示一些样本值
                    if tensor.numel() > 0:
                        sample_values = tensor.flatten()[:10].tolist()
                        issues.append(f"     样本值: {sample_values}")
            else:
                # 即使类型正确，也检查值是否合理
                if tensor.numel() > 0:
                    min_val = tensor.min().item()
                    max_val = tensor.max().item()
                    if min_val < 0:
                        issues.append(f"  ⚠ {field}: 包含负值 (min={min_val})")
        
        if issues:
            return batch
        elif batch_idx % 100 == 0:
            continue



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