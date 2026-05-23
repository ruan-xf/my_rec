
## 1
原来候选物品是单独处理的，现在和历史一起处理了，但是手动 embeddings 调用忘记撤了  
inputs_embeds = self.albert_classifier.albert.embeddings(inputs_embeds=concated)  

这会有以下后果：  
重复计算：两次 embeddings 调用会导致：  
    token_type embeddings 被加了两次  
    position embeddings 被加了两次  
    LayerNorm 被应用了两次  
    Dropout 被应用了两次（这会改变随机性！）  
语义错误：最终得到的 embedding 并不是你想要的，因为：  
    position embedding 被加了两次 → 位置信息被双重加权  
    Dropout 被应用两次 → 正则化效果过强  
潜在的训练不稳定：由于 Dropout 的随机性，两次应用会导致梯度计算不一致  

## 2
@.venv/Lib/site-packages/transformers/models/albert/modeling_albert.py#409-419  

sequence_output[:, 0] 原来表示的是 [CLS] ，但我  
[item1, item2, ..., item128]  →  position 0 = 第一个物品的特征  

拿到的是用户行为序列中第一个物品经过 Transformer 编码后的表示。  


虽然自注意力机制会让 position 0 也 gather 到其他位置的信息（某种程度上起到了聚合作用），但它同时还得保留自身作为"第一个物品"的信息，不是纯粹为分类服务的。  


几个可选方案  
- 保持现状 — 不是不能用，很多 transformer 推荐模型也是取某个位置的输出，效果还行
- 改取最后一个位置 sequence_output[:, -1] — "当前行为"导向，跟行为序列的直觉更接近（类似 SASRec）
- 改成 mean pooling — sequence_output.mean(dim=1)，平均所有位置，跟具体位置无关，更鲁棒
- 手动拼接可学习的 [CLS] token — 在 concated 前面 concat 一个可学习向量