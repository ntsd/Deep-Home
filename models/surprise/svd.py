import pandas as pd

import sys
sys.path.append('./')
from features import features
from metrics import average_precision

from collections import defaultdict
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split

train = pd.read_csv('./input/train.csv', parse_dates=['datetime'])
test = pd.read_csv('./input/test.csv')

# to remove usercode not in test
# user_test = set(test['userCode'].values)
# train = train[train['userCode'].isin(user_test)]
# print(len(train), len(test))

# count userCode feature
gp, name_col = features.count_duplicate(train, group_cols=['userCode', 'project_id'])
train = train.merge(gp, on=['userCode', 'project_id'], how='left')
# print(name_col, train.head(), (1, train[name_col].max()))

#drop duplicate
train = train.drop_duplicates(['userCode','project_id'],keep= 'last')
print(len(train), len(test))

#to scale
scale = train[name_col].max()
train[name_col] = train[name_col].apply(lambda x: x/scale)
print('max: ',train[name_col].max())

reader = Reader(rating_scale=(0, 1))

trainset = Dataset.load_from_df(train[["userCode", "project_id", name_col]], reader)
trainset = trainset.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

accuracy.rmse(predictions, verbose=True)

def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

top_n = get_top_n(predictions, n=7)

#evaluate
actual_list = [[pid] for pid in test['project_id'].values]
predicted_list = [[project_id for (project_id, _) in top_n[uid]]for uid in test['userCode'].values]

print(average_precision.mapk(actual_list, predicted_list))

# Print the recommended items for each user
# for uid, ratings in top_n.items():
#     print(uid, [(project_id, rating) for (project_id, rating) in ratings])
