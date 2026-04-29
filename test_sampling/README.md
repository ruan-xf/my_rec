

新的采样策略：类别和物品都会进行采样，因为要体现模型正确预测用户偏好的程度，非真实的都打为0了模型能学到个啥，所以将以[0,1]的分数表示  

```
类目输入	item输入	期望模型输出	训练信号
真实类目	真实物品	1.0	这是对的
<pad>	真实物品	1.0	不提供物品类别信息，但模型了解物品所属类别或能从历史中正确推断
真实类目	同类别错误物品	a1	在正确类目下选错了物品
<pad>	同类别错误物品	a1	不提供物品类别信息，模型推断出正确类目但选错了物品
真实类目	不同类别错误物品	a2	无视类目信号,推荐了不相关的物品
<pad>	不同类别错误物品	a2	不提供物品类别信息，模型推断出的类目与真实不符
非真实类目	真实物品	0	类目错了
非真实类目	错误物品	0.0
```

目的：模型要从类目到具体物品进行预测，保证从用户历史中准确把握用户偏好，即使模型不能预测准物品，也应该能预测到类别及与类别相关物品  

<pad> 类目情况下：  
- 模型需要从用户历史序列中推断正确的类目
- 如果推断的类目正确，则按照真实类目的标准评分
- 这测试了模型的类目推理能力  


- a1 0.8
- a2 0.4
- 其他 0

a1 评分情况（真实类目/<pad> + 同类别错误物品）：  
在正确推断的类目下选错了物品  
缺乏"物品-类目"的细粒度匹配能力  

a2 评分情况（真实类目/<pad> + 不同类别错误物品）：  
缺乏"物品-类目"的静态知识  
这是一种知识性错误（物品A不属于类目X）  


类目错误的情况  
没从历史序列中推断对用户下一个想交互的类目  
这是一种推理错误（历史→未来的预测失败）  


a1→a2 距离：0.4（惩罚异类错误）  
a2→0 距离：0.4（保持区分度）  
距离均匀，模型能清楚区分"类目正确"和"类目错误"  


验证、测试不采样，那么需要新指标：  
MSE/MAE：直接衡量模型输出与1.0的距离  


新的数据如所示：  
```
[{'item_seq': [{'behavior_type': 'cart',
    'category_id': '5053508',
    'item_id': '2117421'},
   {'behavior_type': 'pv', 'category_id': '1207887', 'item_id': '4705452'},
   {'behavior_type': 'pv', 'category_id': '2979796', 'item_id': '1047458'},
   {'behavior_type': 'pv', 'category_id': '1207887', 'item_id': '4705452'},
   {'behavior_type': 'pv', 'category_id': '2939262', 'item_id': '458480'},
   {'behavior_type': 'pv', 'category_id': '4284875', 'item_id': '1210056'},
   {'behavior_type': 'pv', 'category_id': '4834913', 'item_id': '1752559'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '1570651'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '4633553'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '2684269'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '3096591'},
   {'behavior_type': 'pv', 'category_id': '4533189', 'item_id': '2477822'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '1677233'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '4465071'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2855032'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2205132'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '4026092'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '5109118'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2046206'},
   {'behavior_type': 'pv', 'category_id': '4533189', 'item_id': '3929240'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '3052391'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '4055348'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '4544551'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2016895'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2202800'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '1677233'},
   {'behavior_type': 'pv', 'category_id': '1299190', 'item_id': '3816406'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2840861'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '1992406'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2016895'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '2684269'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '3116095'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '5009111'},
   {'behavior_type': 'pv', 'category_id': '3528894', 'item_id': '1049416'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '377422'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '223488'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '68929'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '2821860'},
   {'behavior_type': 'cart', 'category_id': '1299190', 'item_id': '2867965'},
   {'behavior_type': 'cart', 'category_id': '1884322', 'item_id': '4643821'},
   {'behavior_type': 'pv', 'category_id': '4565874', 'item_id': '380675'},
   {'behavior_type': 'pv', 'category_id': '4565874', 'item_id': '1594680'},
   {'behavior_type': 'pv', 'category_id': '1343555', 'item_id': '4483089'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2930346'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '3597631'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '1453152'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '3116095'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '808263'},
   {'behavior_type': 'cart', 'category_id': '1573426', 'item_id': '1992406'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '3839802'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2051036'},
   {'behavior_type': 'buy', 'category_id': '1573426', 'item_id': '1992406'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2051036'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '1992406'},
   {'behavior_type': 'pv', 'category_id': '2939262', 'item_id': '458480'},
   {'behavior_type': 'pv', 'category_id': '2939262', 'item_id': '1559496'},
   {'behavior_type': 'pv', 'category_id': '2939262', 'item_id': '2155825'},
   {'behavior_type': 'cart', 'category_id': '2982027', 'item_id': '4926670'},
   {'behavior_type': 'cart', 'category_id': '1573426', 'item_id': '1599741'},
   {'behavior_type': '<pad>', 'category_id': '4159072', 'item_id': '240836'}],
  'label': 0.8},
 {'item_seq': [{'behavior_type': 'cart',
    'category_id': '5053508',
    'item_id': '2117421'},
   {'behavior_type': 'pv', 'category_id': '1207887', 'item_id': '4705452'},
   {'behavior_type': 'pv', 'category_id': '2979796', 'item_id': '1047458'},
   {'behavior_type': 'pv', 'category_id': '1207887', 'item_id': '4705452'},
   {'behavior_type': 'pv', 'category_id': '2939262', 'item_id': '458480'},
   {'behavior_type': 'pv', 'category_id': '4284875', 'item_id': '1210056'},
   {'behavior_type': 'pv', 'category_id': '4834913', 'item_id': '1752559'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '1570651'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '4633553'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '2684269'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '3096591'},
   {'behavior_type': 'pv', 'category_id': '4533189', 'item_id': '2477822'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '1677233'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '4465071'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2855032'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2205132'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '4026092'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '5109118'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2046206'},
   {'behavior_type': 'pv', 'category_id': '4533189', 'item_id': '3929240'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '3052391'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '4055348'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '4544551'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2016895'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2202800'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '1677233'},
   {'behavior_type': 'pv', 'category_id': '1299190', 'item_id': '3816406'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2840861'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '1992406'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2016895'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '2684269'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '3116095'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '5009111'},
   {'behavior_type': 'pv', 'category_id': '3528894', 'item_id': '1049416'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '377422'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '223488'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '68929'},
   {'behavior_type': 'pv', 'category_id': '4095810', 'item_id': '2821860'},
   {'behavior_type': 'cart', 'category_id': '1299190', 'item_id': '2867965'},
   {'behavior_type': 'cart', 'category_id': '1884322', 'item_id': '4643821'},
   {'behavior_type': 'pv', 'category_id': '4565874', 'item_id': '380675'},
   {'behavior_type': 'pv', 'category_id': '4565874', 'item_id': '1594680'},
   {'behavior_type': 'pv', 'category_id': '1343555', 'item_id': '4483089'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2930346'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '3597631'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '1453152'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '3116095'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '808263'},
   {'behavior_type': 'cart', 'category_id': '1573426', 'item_id': '1992406'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '3839802'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2051036'},
   {'behavior_type': 'buy', 'category_id': '1573426', 'item_id': '1992406'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '2051036'},
   {'behavior_type': 'pv', 'category_id': '1573426', 'item_id': '1992406'},
   {'behavior_type': 'pv', 'category_id': '2939262', 'item_id': '458480'},
   {'behavior_type': 'pv', 'category_id': '2939262', 'item_id': '1559496'},
   {'behavior_type': 'pv', 'category_id': '2939262', 'item_id': '2155825'},
   {'behavior_type': 'cart', 'category_id': '2982027', 'item_id': '4926670'},
   {'behavior_type': 'cart', 'category_id': '1573426', 'item_id': '1599741'},
   {'behavior_type': '<pad>', 'category_id': '888921', 'item_id': '4588808'}],
  'label': 0.0}]

{'behavior_type': tensor([[6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
          8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8,
          6, 8, 8, 5, 8, 8, 8, 8, 8, 6, 6, 0],
         [6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
          8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8,
          6, 8, 8, 5, 8, 8, 8, 8, 8, 6, 6, 0]]),
 'category_id': tensor([[8124,  416, 3949,  416, 3871, 6632, 7682, 1126, 1126, 6254, 6254, 7112,
          6254, 1126, 1126, 1126, 6254, 1126, 1126, 7112, 1126, 1126, 6254, 1126,
          1126, 6254,  583, 1126, 1126, 1126, 6254, 1126, 1126, 5097, 1126, 1126,
          1126, 6254,  583, 1740, 7178, 7178,  657, 1126, 1126, 1126, 1126, 1126,
          1126, 1126, 1126, 1126, 1126, 1126, 3871, 3871, 3871, 3953, 1126, 6377],
         [8124,  416, 3949,  416, 3871, 6632, 7682, 1126, 1126, 6254, 6254, 7112,
          6254, 1126, 1126, 1126, 6254, 1126, 1126, 7112, 1126, 1126, 6254, 1126,
          1126, 6254,  583, 1126, 1126, 1126, 6254, 1126, 1126, 5097, 1126, 1126,
          1126, 6254,  583, 1740, 7178, 7178,  657, 1126, 1126, 1126, 1126, 1126,
          1126, 1126, 1126, 1126, 1126, 1126, 3871, 3871, 3871, 3953, 1126, 9057]]),
 'item_id': tensor([[ 877877, 2912558,   37265, 2912558, 2817811,  164864,  591256,  448099,
          2856097, 1323763, 1647390, 1161685,  532069, 2723702, 1457464,  946876,
          2378618, 3229639,  821857, 2302267, 1612554, 2401745, 2786063,  798872,
           945053,  532069, 2213411, 1446411,  779668,  798872, 1323763, 1662662,
          3151275,   38864, 2180168,  970349, 3408045, 1431493, 1467530, 2864174,
          2205813,  467002, 2737752, 1516611, 2041807,  355956, 1662662, 3501464,
           779668, 2231917,  825706,  779668,  825706,  779668, 2817811,  439387,
           908068, 3086502,  470968, 1107021],
         [ 877877, 2912558,   37265, 2912558, 2817811,  164864,  591256,  448099,
          2856097, 1323763, 1647390, 1161685,  532069, 2723702, 1457464,  946876,
          2378618, 3229639,  821857, 2302267, 1612554, 2401745, 2786063,  798872,
           945053,  532069, 2213411, 1446411,  779668,  798872, 1323763, 1662662,
          3151275,   38864, 2180168,  970349, 3408045, 1431493, 1467530, 2864174,
          2205813,  467002, 2737752, 1516611, 2041807,  355956, 1662662, 3501464,
           779668, 2231917,  825706,  779668,  825706,  779668, 2817811,  439387,
           908068, 3086502,  470968, 2820963]]),
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
 'labels': tensor([0.8000, 0.0000])}
```