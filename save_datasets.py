import data_util, pyarrow as pa, pyarrow.parquet as pq, os


def save_dataset_to_parquet(ds, out_parquet, batch_size=10000):
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    writer = None
    total = 0
    for batch in ds.iter(batch_size):
        table = pa.Table.from_pydict(batch)
        if writer is None:
            writer = pq.ParquetWriter(out_parquet, table.schema)
        writer.write_table(table)
        total += len(next(iter(batch.values())))
        print(f'  total={total}')
    writer.close()
    print(f'  done: {total} rows, size={os.path.getsize(out_parquet) / 1024 / 1024:.1f} MB')


def main():
    out_dir = 'data/processed/hf_saved'
    os.makedirs(out_dir, exist_ok=True)

    ds_setting = data_util.DatasetSetting(per_eval_size=20, use_sample=False)

    # print('saving train...')
    # save_dataset_to_parquet(
    #     ds_setting.train_dataset,
    #     f'{out_dir}/train.parquet',
    # )

    print('saving eval...')
    eval_dataset = data_util.IterableDataset.from_generator(
        lambda: ds_setting.generate_with_neg('eval')
    )
    save_dataset_to_parquet(
        eval_dataset,
        f'{out_dir}/eval.parquet',
    )

    # print('saving test...')
    # save_dataset_to_parquet(
    #     ds_setting.test_dataset,
    #     f'{out_dir}/test.parquet',
    # )

    print('all done!')


if __name__ == '__main__':
    main()