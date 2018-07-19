import pandas as pd
import scipy.sparse as sparse
import random;random.seed(44)
import numpy as np;np.random.seed(44)
import implicit
import logging

import sys
sys.path.append('./')
from features import features
from metrics import average_precision

import tqdm
from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (AnnoyAlternatingLeastSquares, FaissAlternatingLeastSquares,
                                      NMSLibAlternatingLeastSquares)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)
from implicit.datasets.lastfm import get_lastfm

# maps command line model argument to class name
MODELS = {"als":  AlternatingLeastSquares,
          "nmslib_als": NMSLibAlternatingLeastSquares,
          "annoy_als": AnnoyAlternatingLeastSquares,
          "faiss_als": FaissAlternatingLeastSquares,
          "tfidf": TFIDFRecommender,
          "cosine": CosineRecommender,
          "bpr": BayesianPersonalizedRanking,
          "bm25": BM25Recommender}

model_name = 'als'

def get_model(model_name):
    model_class = MODELS.get(model_name)
    if not model_class:
        raise ValueError("Unknown Model '%s'" % model_name)

    # some default params
    if issubclass(model_class, AlternatingLeastSquares):
        params = {'factors': 64, 'dtype': np.float32}
    elif model_name == "bm25":
        params = {'K1': 100, 'B': 0.5}
    elif model_name == "bpr":
        params = {'factors': 63}
    else:
        params = {}

    return model_class(**params)

is_test = 1
if is_test:
    train = pd.read_csv('./input/train_small.csv')
    test = pd.read_csv('./input/test_small.csv')
else:
    train = pd.read_csv('./input/userLog_201801_201802_for_participants.csv', delimiter=';')
    test = pd.read_csv('./input/testing_users.csv', delimiter=';')

#todo replace with weight
# count feature
gp, rating = features.count_duplicate(train, group_cols=['userCode', 'project_id'])
# rating='interacted'
# train[rating] = 1

#drop duplicate
train = train.drop_duplicates(['userCode','project_id'],keep='last')
print(len(train), len(test))

train = train.merge(gp, on=['userCode', 'project_id'], how='left')
del gp

# df_train_pivot = (df_train_indexed.pivot(index = 'userCode', columns = 'project_id', values = rating)
#                                   .fillna(0))
# print(df_train_pivot.head(3))

# df_train_matrix = df_train_pivot.values
# print(df_train_matrix.shape)

# df_train_indexed = df_train_indexed.set_index('userCode')
# df_train_indexed.head(3)

# sort value
train = train.sort_values(['userCode', 'project_id'])
# print(train)


# Get unique user 
users = list(np.sort(train['userCode'].unique()))

# Get unique project
projects = list(np.sort(train['project_id'].unique()))

# get rating
rating_list = train[rating].tolist()

# Get the associated row/column indices
rows = train.userCode.astype(pd.api.types.CategoricalDtype(categories = users)).cat.codes
cols = train.project_id.astype(pd.api.types.CategoricalDtype(categories = projects)).cat.codes

# print(len(users), len(user_cat), rows.shape)

# create sparse matrix from data
visit_sparse = sparse.csr_matrix((rating_list, (rows, cols)), shape=(len(users), len(projects)), dtype=np.float32).T.tocsr()

del rows
del cols
del train

# use to check metrix
print(len(users))
print(len(projects))
print(visit_sparse.shape)
# for i in range(len(users)):
#     print(users[i], end=' ')
#     for j in range(len(projects)):
#         if visit_sparse[i, j]!=0:
#             print(visit_sparse[i, j], end=' ')
#     print()

model = get_model(model_name)

# if we're training an ALS based model, weight input for last.fm
# by bm25
if issubclass(model.__class__, AlternatingLeastSquares):
    # lets weight these models by bm25weight.
    logging.debug("weighting matrix by bm25_weight")
    visit_sparse = bm25_weight(visit_sparse, K1=100, B=0.8)

    # also disable building approximate recommend index
    model.approximate_similar_items = False

model.fit(visit_sparse)

visit_csr = visit_sparse.T.tocsr()

# recom = model.recommend(userid=0, user_items=visit_csr, N=7)
# print(recom)

predicted_list = []

with tqdm.tqdm(total=len(test)) as progress:
    for uid in test['userCode']:
        recom = model.recommend(userid=users.index(uid), user_items=visit_csr, N=7)
        predicted_list.append([projects[pindex] for pindex, _ in recom])
        progress.update(1)

evaluate = is_test
if evaluate:
    actual_list = [[pid] for pid in test['project_id'].values]
    print('%.10f'%average_precision.mapk(actual_list, predicted_list, k=7))

to_csv = not is_test
if to_csv:
    test['project_id'] = [' '.join(map(str, pre)) for pre in predicted_list]
    test[['userCode','project_id']].to_csv('submission_{}.csv'.format(model_name), index=False)
# 0.0000319489