
import polars as pl
import transformers
from transformers import AutoTokenizer
from datasets import Dataset

df = pl.scan_parquet('../../amazon-book/data/processed/train.parquet', low_memory=True).head(200).collect()
texts = df['item_seq'].list.eval(pl.element().cast(str)).list.join(' ').to_list()

pl_col = pl.col('item_seq')
(
    df
    .lazy()
    .select(pl_col.cast(str).str.join(' '))
    .collect()
    .to_series().to_list()
)



ds = Dataset.from_dict({'text': texts})
tokenizer: transformers.PreTrainedTokenizerFast = AutoTokenizer.from_pretrained('albert-base-v2')

tokenized_ds = ds.map(
    lambda batch: tokenizer(batch['text'], truncation=True),
    batched=True
)


# args = TrainingArguments(
#     per_device_train_batch_size=5,
#     per_device_eval_batch_size=2,
#     # eval_strategy='steps',
#     # eval_steps=0.3,
#     # logging_steps=0.3,
#     max_steps=20,
#     save_strategy='no',
#     report_to='none'
# )


# trainer = Trainer(
#     args=args,
#     # train_dataset=tokenized_ds,
#     eval_dataset=ds.map(
#         lambda batch: tokenizer(batch['text']),
#         batched=True
#     ),
#     processing_class=tokenizer,
#     model_init=lambda : AutoModel.from_pretrained('albert-base-v2')
# )


# seq_lens = [[], []]
# for batch in trainer.get_eval_dataloader():
#     seq_lens[False].append(batch.input_ids.shape[1])

# trainer.eval_dataset = tokenized_ds
# for batch in trainer.get_eval_dataloader():
#     seq_lens[True].append(batch.input_ids.shape[1])

# len_df = pd.DataFrame(seq_lens).T
# len_df[len_df[0] != len_df[1]]
# # 0	1
# # 66	1681	512

# tokenizer.model_max_length
# # 512

# "AlbertForMaskedLM"
# "hidden_act": "gelu_new",
# "embedding_size": 128,
# "max_position_embeddings": 512,
# "vocab_size": 30000
# trainer.model.config

# trainer.evaluate()
# RuntimeError: The size of tensor a (1681) must match the size of tensor b (512) at non-singleton dimension 1


# dataloader = trainer.get_eval_dataloader()


# batch = next(iter(dataloader))
# batch

# DataCollatorWithPadding(processing_class)
