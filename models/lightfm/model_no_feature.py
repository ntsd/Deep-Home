import pandas as pd
import numpy as np


import sys
import tqdm
sys.path.append('./')
from features import features

import time
import matplotlib
import matplotlib.pyplot as plt
from lightfm.evaluation import precision_at_k

train = pd.read_csv('./input/train_20000.csv')
test = pd.read_csv('./input/test_20000.csv')

gp, interaction_col_name = features.count_duplicate(train, group_cols=['userCode', 'project_id'])
train = train.drop_duplicates(['userCode','project_id'],keep='last')
train = train.merge(gp, on=['userCode', 'project_id'], how='left');del gp

from lightfm.data import Dataset
from lightfm import LightFM
# build dataset
dataset = Dataset()

unique_project = train['project_id'].drop_duplicates()
unique_user = train['userCode'].drop_duplicates()

user_iterable = (row for row in unique_user)
iteam_iterable = (row for row in unique_project)

# fit dataset
dataset.fit(users=user_iterable,
            items=iteam_iterable
            )

# check shape
num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items: {}.'.format(num_users, num_items))

(train_interactions, weights) = dataset.build_interactions(data=((row['userCode'], row['project_id'], row[interaction_col_name])for index, row in train.iterrows()))
(test_interactions, _) = dataset.build_interactions(data=((row['userCode'], row['project_id'])for index, row in test.iterrows()))
