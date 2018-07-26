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

is_test=1
if is_test:
    train = pd.read_csv('./input/train_10000.csv')
    test = pd.read_csv('./input/test_10000.csv')
else:
    train = pd.read_csv('./input/userLog_201801_201802_for_participants.csv', delimiter=';')
    test = pd.read_csv('./input/testing_users.csv', delimiter=';')

print('train', train.shape)

# drop duplicate
# train = train.drop_duplicates(['userCode','project_id'],keep='last')

# drop train userCode not in test UserCode 
if not is_test:
    unique_test = test['userCode'].unique()
    train = train[train['userCode'].isin(unique_test)]

print('train', train.shape)

from lightfm.data import Dataset
from lightfm import LightFM

# build dataset
dataset = Dataset()

unique_project = train['project_id'].drop_duplicates()
unique_user = train['userCode'].drop_duplicates()

user_iterable = (row for row in unique_user)
iteam_iterable = (row for row in unique_project)

add_item_feature=1
if add_item_feature: # build item feature
    item_feature_df = pd.read_csv('./input/item_feature.csv')
    item_feature_names = ['facility_1', 'facility_2', 'facility_3', 'facility_4', 'facility_5', 'facility_6'] #, 'starting_price_range_1', 'land_size_range_0', 'land_size_range_3', 'facility_3', 'project_interact_range_3', 'unit_functional_space_starting_size_range_1', 'facility_6', 'facility_2', 'facility_5']
    item_feature_df = item_feature_df[['project_id']+item_feature_names]
    # item_feature_df[item_feature_names] = item_feature_df[item_feature_names].replace({0:-1}) # change 0 to -1 for negative
    # item_feature_names = list(item_feature_df)[1:]
    item_feature_df = item_feature_df[item_feature_df['project_id'].isin(unique_project)]
    item_feature_iterable = ((row['project_id'], [feature_name for feature_name in item_feature_names if row[feature_name]==1])for index, row in item_feature_df.iterrows())

add_user_feature=1
if add_user_feature: # build user feature
    user_feature_df = pd.read_csv('./input/user_feature.csv')
    user_feature_names = ['time_interval_20-23', 'time_interval_9-19', 'time_interval_0-8']#, 'pageReferrer_Other_PageReferer', 'userAgent_Other_OS', 'time_interval_9-19']
    user_feature_df = user_feature_df[['userCode']+user_feature_names]
    # user_feature_df[user_feature_names] = user_feature_df[user_feature_names].replace({0:-1})  # change 0 to -1 for negative
    # user_feature_names = list(user_feature_df)[1:]
    user_feature_df = user_feature_df[user_feature_df['userCode'].isin(unique_user)]
    user_feature_iterable = ((row['userCode'], [feature_name for feature_name in user_feature_names if row[feature_name]==1])for index, row in user_feature_df.iterrows())

# fit dataset
dataset.fit(users=user_iterable,
            items=iteam_iterable,
            user_features=user_feature_names if add_user_feature else None,
            item_features=item_feature_names if add_item_feature else None
            )

# check shape
num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items: {}.'.format(num_users, num_items))
_, num_users_feature = dataset.user_features_shape()
_, num_items_feature = dataset.item_features_shape()
print('Num users feature: {}, num_items feature: {}.'.format(num_users_feature, num_items_feature))

(train_interactions, _) = dataset.build_interactions(data=((row['userCode'], row['project_id'])for index, row in train.iterrows()))
if is_test:
    (test_interactions, _) = dataset.build_interactions(data=((row['userCode'], row['project_id'])for index, row in test.iterrows()))

if add_user_feature:
    # build user feature matrix
    user_feature_matrix = dataset.build_user_features(user_feature_iterable, normalize=False)
    # print(user_feature_matrix)

if add_item_feature:
    # build item feature matrix
    item_feature_matrix = dataset.build_item_features(item_feature_iterable, normalize=False)
    # print(item_feature_matrix)

# create Model
epochs = 1#20 # 60
alpha=0.001
num_components=256 # 256 # 512
schedule='adagrad'

warp_model = LightFM(no_components=num_components,
                            loss='warp',
                            learning_schedule=schedule,
                            # user_alpha=alpha,
                            # item_alpha=alpha,
                            random_state=44
                            )

warp_duration = []
warp_pre = []

max_precission = 0
old_pickle_name = ''
for epoch in range(epochs):
    start = time.time()
    if epochs==1:
        warp_model.fit(train_interactions,
                            epochs=20,
                            num_threads=8,
                            item_features=item_feature_matrix if add_item_feature else None,
                            user_features=user_feature_matrix if add_user_feature else None)
    else:
        warp_model.fit_partial(train_interactions,
                            epochs=1,
                            num_threads=8,
                            item_features=item_feature_matrix if add_item_feature else None,
                            user_features=user_feature_matrix if add_user_feature else None)
    time_=time.time() - start
    warp_duration.append(time_)
    if is_test:
        precission=precision_at_k(warp_model, test_interactions, train_interactions=train_interactions, k=7,
                                user_features=user_feature_matrix if add_user_feature else None,
                                item_features=item_feature_matrix if add_item_feature else None
                                ).mean()
    else:
        precission=precision_at_k(warp_model, train_interactions, k=7,
                                user_features=user_feature_matrix if add_user_feature else None,
                                item_features=item_feature_matrix if add_item_feature else None
                                ).mean()
    warp_pre.append(precission)
    print('Fit Model Finish Epoch: {} ACC: {} TIME {}:'.format(epoch, precission, time_))
    # save model checkpoint
    # if precission < max_precission:
    #     break
    if not is_test and precission > max_precission:
        pickle_name = 'warp_model_{}_{}_drop_usercode.pickle'.format(epoch, precission)
        with open(pickle_name, 'wb') as file_:
            pickle.dump(warp_model, file_, protocol=pickle.HIGHEST_PROTOCOL)
        # if old_pickle_name != '':os.remove(old_pickle_name) 
        old_pickle_name = pickle_name
        max_precission = precission

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
            predictions = warp_model.predict(unique_user_list.index(uid),
                                    np.arange(num_project),
                                    user_features=user_feature_matrix if add_user_feature else None,
                                    item_features=item_feature_matrix if add_item_feature else None
                                    )
            top_items = unique_project.iloc[np.argsort(-predictions)]
            top_list = []
            top_n = 0
            for project_id in top_items.values:
                if project_id not in visited_dict[uid]: # todo add ignore project
                    top_list.append(project_id)
                    top_n+=1
                # if top_n >= 7:
                #     break
            predicted_list.append(top_list)
            progress.update(1)

    to_csv = 0
    if to_csv:
        test['project_id'] = [' '.join(map(str, pre)) for pre in predicted_list]
        csv_name = 'submission_{}_{}_{}_with_feature_no_clean_drop_usercode.csv'.format(num_components, alpha, schedule)
        test[['userCode','project_id']].to_csv(csv_name, index=False)

    if is_test:
        actual_list = [[pid] for pid in test['project_id'].values]
        print('%.10f'%average_precision.mapk(actual_list, predicted_list, k=7))

# plot graph
x = np.arange(len(warp_pre))
plt.plot(x, np.array(warp_pre))
plt.legend(['WARP P@K'], loc='upper right')
eval_name = 'eval_{}_{}_{}_with_feature_no_clean_drop_usercode'.format(num_components, alpha, schedule)
plt.savefig('{}.png'.format(eval_name))
plt.clf()
plt.cla()
# eval_df = pd.DataFrame({'WARP_PAK': warp_pre})
# eval_df.to_csv('{}.csv'.format(eval_name), index=False)

x = np.arange(len(warp_duration))
plt.plot(x, np.array(warp_duration))
plt.legend(['WARP duration'], loc='upper right')
time_name = 'time_{}_{}_{}_with_feature_no_clean_drop_usercode'.format(num_components, alpha, schedule)
plt.savefig('{}.png'.format(time_name))
plt.clf()
plt.cla()
# time_df = pd.DataFrame({'WARP_TIME': warp_duration})
# time_df.to_csv('{}.csv'.format(time_name), index=False)
