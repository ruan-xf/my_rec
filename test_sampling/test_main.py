
import main

example_last_item = {'behavior_type': 'pv', 'category_id': 4159072, 'item_id': 3564386}


[
    main.sample_positive(example_last_item, False),
    main.sample_positive(example_last_item, True),
    main.sample_same_category_wrong_item(example_last_item, False),
    main.sample_same_category_wrong_item(example_last_item, True),
    main.sample_diff_category_wrong_item(example_last_item, False),
    main.sample_diff_category_wrong_item(example_last_item, True),
    main.sample_wrong_category_correct_item(example_last_item),
    main.sample_wrong_category_wrong_item(example_last_item)
]


# [{'item': 3564386, 'category': 4159072, 'label': 1.0},
#  {'item': 3564386, 'category': '<pad>', 'label': 1.0},
#  {'item': np.int64(3037490), 'category': 4159072, 'label': 0.8},
#  {'item': np.int64(4341131), 'category': '<pad>', 'label': 0.8},
#  {'item': np.int64(1291189), 'category': 4159072, 'label': 0.4},
#  {'item': np.int64(4853421), 'category': '<pad>', 'label': 0.4},
#  {'item': 3564386, 'category': np.int64(1746357), 'label': 0.0},
#  {'item': np.int64(2185153), 'category': np.int64(880704), 'label': 0.0}]