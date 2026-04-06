
import pandas as pd



unique_items = pd.read_parquet('unique_items.parquet')

unique_items.groupby('item_id').count().value_counts()
# category_id
# 1              4141196
# 2                 1378
# 3                    7
# 4                    2
# Name: count, dtype: int64


ser = unique_items.groupby('item_id').count() > 1
ser = ser.squeeze()

ser.index[ser]
unique_items[unique_items.item_id == 1608]
# 	category_id	item_id
# 486393	3886822	1608
# 2699115	4449178	1608
