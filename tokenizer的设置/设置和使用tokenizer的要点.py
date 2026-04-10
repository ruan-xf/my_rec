
import pandas as pd
import polars as pl
import transformers

df = pd.read_csv('../../example.csv')

book_ser = pd.concat([
    df.book_id_seq.str.split('|').explode(),
    df.book_id,
]).astype(str).drop_duplicates(ignore_index=True)


from tokenizers import models, pre_tokenizers, trainers, Tokenizer
from transformers import PreTrainedTokenizerFast, AlbertTokenizerFast

tokenizer = Tokenizer(models.WordLevel())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 这样的special tokens编码是符合预期的，包装类的train也会进行同样的设置，可以此为基进一步寻找包装类的special tokens的编码问题
# special_tokens = ['[UNK]', "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
# trainer = trainers.WordLevelTrainer(vocab_size=1400, special_tokens=special_tokens)
# tokenizer.train_from_iterator([book_ser.to_string(index=False)], trainer)

# 虽然tokenier本身也有save方法，但不好用就是了
# # tokenizer.model.save('example_tokenizer')
# tokenizer.save('tokenizer.json')

# 必须要提前add以保证编码
tokenizer.add_special_tokens(['<pad>', '<unk>', '[CLS]', '[SEP]', '[MASK]'])
hf_tokenizer = AlbertTokenizerFast(
    tokenizer_object=tokenizer,
    **{"model_max_length": 512},
)

# 完全可以设置一个足够大的数，但有OverflowError又不能使用hf VERY_LARGE_INTEGER那么大的
hf_tokenizer: AlbertTokenizerFast = hf_tokenizer.train_new_from_iterator(
    [book_ser.to_string(index=False)],
    vocab_size=int(1e16),
)

# 验证得到的词表大小
# assert len(hf_tokenizer.all_special_ids) + len(book_ser) == len(hf_tokenizer.get_vocab())
# hf_tokenizer.vocab_size

# 虽然tokenizer本身不能pad, truncate，但经包装后是可以的，也就无需额外设置了
# hf_tokenizer._tokenizer.padding, hf_tokenizer._tokenizer.truncation
# (None, None)

# hf_tokenizer.batch_decode(
#     hf_tokenizer.batch_encode_plus(
#         ['94122\n129792\n  4157', '669792\n  4157'],
#         padding=True,
#     ).input_ids,
# )
# ['94122 129792 4157', '<unk> 4157 <pad>']

hf_tokenizer.save_pretrained('tokenizers/example')
