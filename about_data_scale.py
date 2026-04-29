import data_util

# df = data_util.read_sample('train')
df = data_util.read_full()

interaction_count = df['item_seq_len'].sum()
n_users = len(df)
# (98914533, 987984)

df['item_seq_len'].eq(1).sum()
# 92



import sampling

n_categorys = len(sampling.all_categories)
n_items = len(sampling.all_items)

# (9435, 4142583)