
TypeError: argument 'vocab': 'dict' object cannot be converted to 'Sequence'  

修改 tokenizer_config.json 中的 "tokenizer_class"  
"AlbertTokenizerFast" -> "PreTrainedTokenizerFast"  


在 collate_fn 中出错，那还是可以用 test_collate_and_model.py 来测试  
只是 token_type_ids 不默认返回了，设置一下 return_token_type_ids  


tensorboard的输出不能使用 args.logging_dir 了，需要设置环境变量 TENSORBOARD_LOGGING_DIR  
> Deprecated and will be removed in v5.2. Set env var `TENSORBOARD_LOGGING_DIR` instead.


self.all_tied_weights_keys = {}  # transformers 5.5+ 要求  

NotebookProgressCallback RuntimeError: on_train_begin must be called before on_evaluate  


