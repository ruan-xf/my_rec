
import os
from pathlib import Path

# import sys
# sys.path.append("..")


# 切换到项目根目录，确保相对路径正确
project_root = Path(__file__).parent.parent.resolve()
os.chdir(project_root)

import data_util

#
# 不同类别错误物品

df = data_util.read_full()


# example_df = df.head()['item_seq'].explode().struct.unnest()
# category_to_items = example_df.group_by('category_id').agg('item_id').to_pandas().set_index('category_id')['item_id']
# category_to_items[753984]



os.chdir(os.path.dirname(__file__))

# example_df = df.head()['item_seq'].explode().struct.unnest().group_by('category_id').agg('item_id')
# example_df.write_parquet('example.parquet')

unique_items = (
    df['item_seq']
    .explode().struct.unnest()['category_id', 'item_id']
    .unique()
)

unique_items.write_parquet('unique_items.parquet')
(
    unique_items
    .group_by('category_id')
    .agg('item_id')
    .write_parquet('category_items.parquet')
)
