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

# build item feature
item_feature_df = pd.read_csv('./input/item_feature.csv')
# drop unimportance
item_feature_df = item_feature_df.drop(['district_id_1302.0', 'district_id_1032.0', 'district_id_2401.0', 'district_id_1010.0', 'province_id_21.0', 'district_id_1206.0', 'province_id_24.0', 'province_id_14.0', 'district_id_1202.0', 'district_id_7306.0', 'district_id_1205.0', 'district_id_4001.0', 'district_id_5001.0', 'district_id_1027.0', 'district_id_2101.0', 'province_id_83.0', 'district_id_1036.0', 'district_id_8303.0', 'district_id_5013.0', 'district_id_5014.0', 'district_id_1021.0', 'province_id_30.0', 'district_id_1406.0', 'district_id_1017.0', 'district_id_1042.0', 'district_id_9201.0', 'district_id_1034.0', 'district_id_2009.0', 'province_id_90.0', 'district_id_1006.0', 'district_id_1025.0', 'district_id_7503.0', 'district_id_1301.0', 'province_id_84.0', 'unit_type_id_3.0', 'district_id_1401.0', 'district_id_1009.0', 'district_id_1038.0', 'district_id_1105.0', 'district_id_1411.0', 'district_id_1102.0', 'district_id_1044.0', 'district_id_5015.0', 'district_id_2002.0', 'district_id_2102.0', 'district_id_1040.0', 'district_id_1023.0', 'district_id_8301.0', 'district_id_9011.0', 'district_id_8302.0', 'district_id_7301.0', 'district_id_2105.0', 'unit_type_id_5.0', 'district_id_8001.0', 'province_id_80.0', 'district_id_2108.0', 'district_id_1033.0', 'district_id_2404.0', 'district_id_1026.0', 'district_id_2405.0', 'province_id_76.0', 'district_id_7707.0', 'district_id_1035.0', 'district_id_8401.0', 'district_id_1049.0', 'province_id_41.0', 'district_id_3401.0', 'district_id_1050.0', 'district_id_1024.0', 'province_id_86.0', 'province_id_34.0', 'province_id_16.0', 'district_id_1041.0', 'district_id_1004.0', 'district_id_1307.0', 'district_id_8601.0', 'district_id_3021.0', 'province_id_92.0', 'district_id_1045.0', 'province_id_52.0', 'district_id_3415.0', 'district_id_1012.0', 'province_id_57.0', 'district_id_7402.0', 'unit_type_id_6.0', 'district_id_1019.0', 'district_id_1901.0', 'district_id_1048.0', 'district_id_1015.0', 'district_id_6701.0', 'district_id_1039.0', 'province_id_70.0', 'district_id_1018.0', 'province_id_19.0', 'district_id_7201.0', 'district_id_1007.0', 'province_id_65.0', 'province_id_71.0', 'district_id_9001.0', 'district_id_7403.0', 'district_id_1903.0', 'district_id_8402.0', 'district_id_1414.0', 'district_id_7101.0', 'district_id_6501.0', 'district_id_4701.0', 'province_id_51.0', 'district_id_7001.0', 'province_id_32.0', 'district_id_9014.0', 'district_id_1001.0', 'district_id_2103.0', 'province_id_75.0', 'district_id_5701.0', 'district_id_7307.0', 'district_id_1037.0', 'district_id_1028.0', 'district_id_6201.0', 'district_id_7005.0', 'district_id_2006.0', 'province_id_60.0', 'district_id_1014.0', 'province_id_62.0', 'district_id_1016.0', 'district_id_5007.0', 'province_id_72.0', 'district_id_1031.0', 'province_id_44.0', 'district_id_1305.0', 'province_id_77.0', 'district_id_7305.0', 'district_id_4301.0', 'district_id_7604.0', 'district_id_7601.0', 'district_id_5501.0', 'province_id_26.0', 'district_id_1020.0', 'district_id_4501.0', 'province_id_31.0', 'district_id_5101.0', 'province_id_47.0', 'district_id_7701.0', 'district_id_2604.0', 'district_id_2011.0', 'district_id_8417.0', 'province_id_27.0', 'district_id_6001.0', 'district_id_2409.0', 'district_id_6711.0', 'district_id_4401.0', 'province_id_25.0', 'district_id_5201.0', 'district_id_3201.0', 'province_id_67.0', 'district_id_4404.0', 'district_id_5401.0', 'province_id_43.0', 'district_id_2502.0', 'district_id_8408.0', 'district_id_5702.0', 'district_id_5605.0', 'district_id_3101.0', 'district_id_3007.0', 'district_id_1601.0', 'district_id_2701.0', 'district_id_7605.0', 'district_id_4005.0', 'province_id_81.0', 'province_id_55.0', 'district_id_7501.0', 'province_id_63.0', 'district_id_1412.0', 'province_id_37.0', 'province_id_56.0', 'province_id_22.0', 'district_id_8415.0', 'district_id_1008.0', 'district_id_7302.0', 'district_id_8101.0', 'district_id_2706.0', 'district_id_2501.0', 'district_id_7303.0', 'district_id_3707.0', 'district_id_1911.0', 'province_id_45.0', 'district_id_5012.0', 'district_id_5011.0', 'province_id_42.0', 'province_id_54.0', 'district_id_6306.0', 'district_id_2406.0', 'district_id_1909.0', 'district_id_9016.0', 'district_id_8404.0', 'district_id_1304.0', 'district_id_7204.0', 'district_id_1902.0', 'district_id_9304.0', 'district_id_2403.0', 'district_id_5709.0', 'district_id_7103.0', 'district_id_5601.0', 'district_id_7706.0', 'district_id_2201.0', 'district_id_7106.0', 'district_id_3411.0', 'district_id_8604.0', 'district_id_3018.0', 'district_id_3009.0', 'province_id_93.0', 'district_id_5514.0', 'district_id_4201.0', 'district_id_4505.0', 'district_id_5106.0', 'district_id_4019.0', 'district_id_3701.0', 'province_id_82.0', 'province_id_23.0', 'district_id_2307.0', 'district_id_1002.0', 'district_id_5710.0', 'district_id_7708.0', 'district_id_8205.0', 'district_id_5025.0'] ,axis=1)

item_feature_names = list(item_feature_df)[1:]
item_feature_df = item_feature_df[item_feature_df['project_id'].isin(unique_project)]
item_feature_iterable = ((row['project_id'], {feature_name: row[feature_name] for feature_name in item_feature_names})for index, row in item_feature_df.iterrows())

# build user feature
user_feature_df = pd.read_csv('./input/user_feature.csv')
user_feature_df = user_feature_df.drop(['userAgent_Other_OS'], axis=1)
user_feature_names = list(user_feature_df)[1:]
user_feature_df = user_feature_df[user_feature_df['userCode'].isin(unique_user)]
user_feature_iterable = ((row['userCode'], {feature_name: row[feature_name] for feature_name in user_feature_names})for index, row in user_feature_df.iterrows())

# fit dataset
dataset.fit(users=user_iterable,
            items=iteam_iterable,
            user_features=user_feature_names,
            item_features=item_feature_names
            )

# check shape
num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items: {}.'.format(num_users, num_items))
_, num_users_feature = dataset.user_features_shape()
_, num_items_feature = dataset.item_features_shape()
print('Num users feature: {}, num_items feature: {}.'.format(num_users_feature, num_items_feature))

# build user feature matrix
user_feature_matrix = dataset.build_user_features(user_feature_iterable, normalize=True)

# build item feature matrix
item_feature_matrix = dataset.build_item_features(item_feature_iterable, normalize=True)

(train_interactions, _) = dataset.build_interactions(data=((row['userCode'], row['project_id'])for index, row in train.iterrows()))
if is_test:
    (test_interactions, _) = dataset.build_interactions(data=((row['userCode'], row['project_id'])for index, row in test.iterrows()))

# create Model
epochs = 70 # 60
alpha=0.001
num_components=256 # 256 # 512
schedule='adagrad'

warp_model = LightFM(no_components=num_components,
                            loss='warp',
                            learning_schedule=schedule,
                            user_alpha=alpha,
                            item_alpha=alpha,
                            # random_state=44
                            )

warp_duration = []
warp_pre = []

max_precission = 0
old_pickle_name = ''
for epoch in range(epochs):
    start = time.time()
    warp_model.fit_partial(train_interactions,
                        epochs=1,
                        num_threads=8,
                        item_features=item_feature_matrix,
                        user_features=user_feature_matrix,)
    time_=time.time() - start
    warp_duration.append(time_)
    if is_test:
        precission=precision_at_k(warp_model, test_interactions, train_interactions=train_interactions,
         k=7, user_features=user_feature_matrix, item_features=item_feature_matrix).mean()
    else:
        precission=precision_at_k(warp_model, train_interactions, k=7,
         user_features=user_feature_matrix, item_features=item_feature_matrix).mean()
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
                                    user_features=user_feature_matrix,
                                    item_features=item_feature_matrix
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

    to_csv = 1
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
eval_df = pd.DataFrame({'WARP_PAK': warp_pre})
eval_df.to_csv('{}.csv'.format(eval_name), index=False)

x = np.arange(len(warp_duration))
plt.plot(x, np.array(warp_duration))
plt.legend(['WARP duration'], loc='upper right')
time_name = 'time_{}_{}_{}_with_feature_no_clean_drop_usercode'.format(num_components, alpha, schedule)
plt.savefig('{}.png'.format(time_name))
plt.clf()
plt.cla()
time_df = pd.DataFrame({'WARP_TIME': warp_duration})
time_df.to_csv('{}.csv'.format(time_name), index=False)
