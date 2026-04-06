
TypeError: argument 'vocab': 'dict' object cannot be converted to 'Sequence'  

修改 tokenizer_config.json 中的 "tokenizer_class"  
"AlbertTokenizerFast" -> "PreTrainedTokenizerFast"  


在 collate_fn 中出错，那还是可以用 test_collate_and_model.py 来测试  
只是 token_type_ids 不默认返回了，设置一下 return_token_type_ids  

