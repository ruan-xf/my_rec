import os
from pathlib import Path


project_root = Path(__file__).parent.parent.resolve()
os.chdir(project_root)


import data_util


df = data_util.read_sample('train')


df[0, 'item_seq'][-1]
# {'behavior_type': 'pv', 'category_id': 4159072, 'item_id': 3564386}
