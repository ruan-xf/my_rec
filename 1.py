

import polars as pl


df = pl.scan_parquet('../data/raw.parquet', low_memory=True).head()

# pl_col = pl.col('category_id')
# (
#     df
#     .lazy()
#     .select(pl_col.cast(str).str.join(' '))
#     .collect()
#     .to_series().to_list()
# )


df.