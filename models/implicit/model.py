import pandas as pd
import scipy.sparse as sparse
import random;random.seed(44)
import numpy as np;np.random.seed(44)
import implicit

import sys
sys.path.append('./')
from features import features
from metrics import average_precision

train = pd.read_csv('./input/train_small.csv', parse_dates=['datetime'])
test = pd.read_csv('./input/test_small.csv')

#todo replace with weight
# count feature
# gp, name_col = features.count_duplicate(train, group_cols=['userCode', 'project_id']) 
# train = train.merge(gp, on=['userCode', 'project_id'], how='left')

#drop duplicate
train = train.drop_duplicates(['userCode','project_id'],keep='last')
print(len(train), len(test))

#drop column
train = train[['userCode', 'project_id', name_col]]

# sort value
train = train.sort_values(['userCode', 'project_id'])
# print(train)

# Get unique user 
users = list(train['userCode'].unique())

# Get unique project
projects = list(train['project_id'].unique())

# get rating
rating = list(train[name_col])

# Get the associated row/column indices
user_cat = train['userCode'].astype('category', categories=users)
project_cat = train['project_id'].astype('category', categories=projects)
rows = user_cat.cat.codes
cols = project_cat.cat.codes

# print(len(users), len(user_cat), rows.shape)

# create sparse matrix from data
visit_sparse = sparse.csr_matrix((rating, (rows, cols)), shape=(len(users), len(projects)), dtype=np.float32).T.tocsr()

# use to check metrix
# print(visit_sparse.shape)
# for i in range(len(users)):
#     for j in range(len(projects)):
#         if visit_sparse[i, j]>1:
#             print(visit_sparse[i, j], train[train['userCode']==users[i]][train['project_id']==projects[j]][name_col])


# Build, fit model and recommend top 7 projects for first user
# model = implicit.als.AlternatingLeastSquares(factors=64, iterations=50) # use_gpu=True

# lets weight these models by bm25weight.
# visit_sparse = implicit.nearest_neighbours.bm25_weight(visit_sparse, K1=100, B=0.8)

# also disable building approximate recommend index
# model.approximate_recommend = False

model = implicit.bpr.BayesianPersonalizedRanking(factors=127)

model.fit(visit_sparse)

visit_csr = visit_sparse.T.tocsr()

recom = model.recommend(userid=0, user_items=visit_csr, N=7)
print(recom)

#evaluate
actual_list = [[pid] for pid in test['project_id'].values]
predicted_list = []

for uid in test['userCode']:
    recom = model.recommend(userid=users.index(uid), user_items=visit_csr, N=7)
    predicted_list.append([pindex for pindex, _ in recom])

# actual_list = [[pid] for pid in test['project_id'].values]
# predicted_list = [[rpid] for rpid in test['project_id'].values]

print(float(average_precision.mapk(actual_list, predicted_list, k=7)))