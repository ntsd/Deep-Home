import pandas as pd
import numpy as np

import sys
import tqdm
sys.path.append('./')
from features import features
from metrics import average_precision

is_test = 1
if is_test:
    train = pd.read_csv('./input/train_tiny.csv')
    test = pd.read_csv('./input/test_tiny.csv')
else:
    train = pd.read_csv('./input/userLog_201801_201802_for_participants.csv', delimiter=';')
    test = pd.read_csv('./input/testing_users.csv', delimiter=';')

from lightfm.data import Dataset
# print([row for row in train['userCode'].unique()])
# build dataset
dataset = Dataset()

unique_project = train['project_id'].drop_duplicates()
unique_user = train['userCode'].drop_duplicates()
# print(type(unique_project))

user_iterable = (row for row in unique_user)
iteam_iterable = (row for row in unique_project)

# build item feature
item_feature_df = pd.read_csv('./input/item_feature.csv')
item_feature_df.drop(['landSize', 'unit_functional_space_starting_size'], axis=1, inplace=True) # drop for only normalize
item_feature_names = list(item_feature_df)[1:]
item_feature_df = item_feature_df[item_feature_df['project_id'].isin(unique_project)]
item_feature_iterable = ((row['project_id'], {feature_name: row[feature_name] for feature_name in item_feature_names})for index, row in item_feature_df.iterrows())

# build user feature
user_feature_df = pd.read_csv('./input/user_feature.csv')
user_feature_names = list(user_feature_df)[1:]
user_feature_df = user_feature_df[user_feature_df['userCode'].isin(unique_user)]
user_feature_iterable = ((row['userCode'], {feature_name: row[feature_name] for feature_name in user_feature_names})for index, row in user_feature_df.iterrows())

dataset.fit(users=user_iterable,
            items=iteam_iterable,
            user_features=user_feature_names,
            item_features=item_feature_names
            )

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items: {}.'.format(num_users, num_items))
_, num_users_feature = dataset.user_features_shape()
_, num_items_feature = dataset.item_features_shape()
print('Num users feature: {}, num_items feature: {}.'.format(num_users_feature, num_items_feature))

user_feature_matrix = dataset.build_user_features(user_feature_iterable, normalize=True)

item_feature_matrix = dataset.build_item_features(item_feature_iterable, normalize=True)

# build interaction
gp, interaction_col_name = features.count_duplicate(train, group_cols=['userCode', 'project_id'])
train = train.drop_duplicates(['userCode','project_id'],keep='last')
train = train.merge(gp, on=['userCode', 'project_id'], how='left');del gp
(train_interactions, weights) = dataset.build_interactions(data=((row['userCode'], row['project_id'], row[interaction_col_name])for index, row in train.iterrows()))

from lightfm import LightFM

model = LightFM(loss='bpr')
model.fit(train_interactions,
        item_features=item_feature_matrix,
        user_features=user_feature_matrix
        )

is_evaluate=0
if is_evaluate:
    from lightfm.evaluation import precision_at_k
    # train_precision = precision_at_k(model, train_interactions, k=7, user_features=user_feature_matrix, item_features=item_feature_matrix).mean()
    # print('Precision: train %.10f.' % train_precision)
    (test_interactions, weights) = dataset.build_interactions(data=((row['userCode'], row['project_id'])for index, row in test.iterrows()))
    test_precision = precision_at_k(model,
                                    test_interactions,
                                    k=7,
                                    user_features=user_feature_matrix,
                                    item_features=item_feature_matrix
                                    ).mean()
    print('Precision: test %.10f.' % test_precision)

is_predict=1
if is_predict:
    num_project = len(unique_project)
    unique_user_list = unique_user.tolist()

    # create visited dict
    visited_dict = train.groupby('userCode')['project_id'].apply(lambda x: list(x)).to_dict()

    predicted_list = []

    with tqdm.tqdm(total=len(test)) as progress:
        for uid in test['userCode'].unique():
            predictions = model.predict(unique_user_list.index(uid),
                                    np.arange(num_project),
                                    user_features=user_feature_matrix,
                                    item_features=item_feature_matrix
                                    )
            top_items = unique_project.iloc[np.argsort(-predictions)]
            top_list = []
            top_n = 0
            for project_id in top_items.values:
                if project_id not in visited_dict[uid]: # todo
                    top_list.append(project_id)
                    top_n+=1
                if top_n >= 7:
                    break
            predicted_list.append(top_list)
            progress.update(1)

    to_csv = 1
    if to_csv:
        test['project_id'] = [' '.join(map(str, pre)) for pre in predicted_list]
        test[['userCode','project_id']].to_csv('submission_{}.csv'.format('lightfm'), index=False)
    # print(top_items.values)
    if is_evaluate:
        actual_list = [[pid] for pid in test['project_id'].values]
        print('%.10f'%average_precision.mapk(actual_list, predicted_list, k=7))