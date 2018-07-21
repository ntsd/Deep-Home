
import pandas as pd
import numpy as np

import os
import sys
import tqdm
import pickle
sys.path.append('./')
from features import features
from metrics import average_precision

import time
import matplotlib
import matplotlib.pyplot as plt
from lightfm.evaluation import precision_at_k

is_test=0
if is_test:
    train = pd.read_csv('./input/train_20000.csv')
    test = pd.read_csv('./input/test_20000.csv')
else:
    train = pd.read_csv('./input/userLog_201801_201802_for_participants.csv', delimiter=';')
    test = pd.read_csv('./input/testing_users.csv', delimiter=';')

print('train', train.shape)

# drop duplicate
train = train.drop_duplicates(['userCode','project_id'],keep='last')

# drop train userCode not in test UserCode todo
# unique_test = test['userCode'].unique()
# train = train[train['userCode'].isin(unique_test)]

print('train', train.shape)

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

# (train_interactions, _) = dataset.build_interactions(data=((row['userCode'], row['project_id'])for index, row in train.iterrows()))
if is_test:
    (test_interactions, _) = dataset.build_interactions(data=((row['userCode'], row['project_id'])for index, row in test.iterrows()))

# load model
model_path = "warp_model_8_0.007098305970430374.pickle"
file_ = open(model_path,'rb')
model = pickle.load(file_)
file_.close()

# predict
is_predict = 1
if is_predict:
    num_project = len(unique_project)
    unique_user_list = unique_user.tolist()
    unique_project_list = unique_project.tolist()

    # create visited dict
    visited_dict = train.groupby('userCode')['project_id'].apply(lambda x: list(x)).to_dict()

    predicted_list = []

    with tqdm.tqdm(total=len(test)) as progress:
        for uid in test['userCode'].unique():
            predictions = model.predict(unique_user_list.index(uid),
                                    np.arange(num_project),
                                    # user_features=user_feature_matrix,
                                    # item_features=item_feature_matrix
                                    )
            top_items = unique_project.iloc[np.argsort(-predictions)]
            top_list = []
            top_n = 0
            for project_id in top_items.values:
                if project_id not in visited_dict[uid]: # todo add ignore project
                    top_list.append(project_id)
                    top_n+=1
                if top_n >= 7:
                    break
            predicted_list.append(top_list)
            progress.update(1)

    to_csv = 1
    if to_csv:
        test['project_id'] = [' '.join(map(str, pre)) for pre in predicted_list]
        csv_name = 'submission_no_feature_no_clean_no_drop_usercode_{}.csv'.format(time.strftime("%Y%m%d-%H%M%S"))
        test[['userCode','project_id']].to_csv(csv_name, index=False)

    if is_test:
        actual_list = [[pid] for pid in test['project_id'].values]
        print('%.10f'%average_precision.mapk(actual_list, predicted_list, k=7))
