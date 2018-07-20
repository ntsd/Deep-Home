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

# for model bpr is not good so we drop first

# the winner is warp, adagrade , more component 128 fast 256 slow is best maybe, more epoch more than 20
# no sample_weight is better than have

for num_components in [128]:# log is not about num_components and epoch [32, 64, 128, 256]
    for schedule in ['adagrad']: # adagrad is better than adadelta ['adagrad','adadelta']
        # num_components = 32 # 128
        epochs = 40 # 40-50
        # schedule = 'adagrad'
        alpha=0.001 # todo can better

        warp_weight_model = LightFM(no_components=num_components,
                            loss='warp',
                            learning_schedule=schedule,
                            user_alpha=alpha,
                            item_alpha=alpha
                            )

        warp_model = LightFM(no_components=num_components,
                            loss='warp',
                            learning_schedule=schedule,
                            user_alpha=alpha,
                            item_alpha=alpha)

        warp_duration = []
        warp_weight_duration = []
        warp_auc = []
        warp_weight_auc = []

        for epoch in range(epochs):
            start = time.time()
            warp_weight_model.fit_partial(train_interactions, epochs=1, num_threads=8)
            warp_weight_duration.append(time.time() - start)
            warp_weight_auc.append(precision_at_k(warp_weight_model, test_interactions, train_interactions=train_interactions, k=7).mean())

        for epoch in range(epochs):
            start = time.time()
            warp_model.fit_partial(train_interactions, epochs=1, num_threads=8)
            warp_duration.append(time.time() - start)
            warp_auc.append(precision_at_k(warp_model, test_interactions, train_interactions=train_interactions, k=7).mean())

        x = np.arange(epochs)
        plt.plot(x, np.array(warp_auc))
        plt.plot(x, np.array(warp_weight_auc))
        plt.legend(['WARP P@K', 'warp_weight P@K'], loc='upper right')
        eval_name = 'eval_{}_{}_{}_weight'.format(num_components, alpha, schedule)
        plt.savefig('{}.png'.format(eval_name))
        plt.clf()
        plt.cla()
        eval_df = pd.DataFrame({'WARP_PAK': warp_auc,
            'warp_weight_PAK': warp_weight_auc
        })
        eval_df.to_csv('{}.csv'.format(eval_name), index=False)

        x = np.arange(epochs)
        plt.plot(x, np.array(warp_duration))
        plt.plot(x, np.array(warp_weight_duration))
        plt.legend(['WARP duration', 'warp_weight duration'], loc='upper right')
        time_name = 'time_{}_{}_{}_weight'.format(num_components, alpha, schedule)
        plt.savefig('{}.png'.format(time_name))
        plt.clf()
        plt.cla()
        time_df = pd.DataFrame({'WARP_TIME': warp_duration,
            'warp_weight_TIME': warp_weight_duration
        })
        time_df.to_csv('{}.csv'.format(time_name), index=False)
