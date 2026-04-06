import polars as pl
import sampling


df = sampling.df

df[df.item_id.apply(len) == 1]
# len = 662 换言之对于这些物品是找不到同类别的其他物品的
