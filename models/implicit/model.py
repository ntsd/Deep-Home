import pandas as pd
import scipy.sparse as sparse
import random;random.seed(44)
import numpy as np;np.random.seed(44)
import implicit

import sys
sys.path.append('./')
from features import features
from metrics import average_precision

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

train = pd.read_csv('./input/train_small.csv')
test = pd.read_csv('./input/test_small.csv')

model_name = 'bpr'

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

#drop column
train = train[['userCode', 'project_id', rating]]

# sort value
train = train.sort_values(['userCode', 'project_id'])
# print(train)

# Get unique user 
users = list(train['userCode'].unique())

# Get unique project
projects = list(train['project_id'].unique())

# get rating
rating_list = list(train[rating])

# Get the associated row/column indices
rows = train['userCode'].astype('category', categories=users).cat.codes
cols = train['project_id'].astype('category', categories=projects).cat.codes

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
    plays = bm25_weight(plays, K1=100, B=0.8)

    # also disable building approximate recommend index
    model.approximate_similar_items = False

model.fit(visit_sparse)

visit_csr = visit_sparse.T.tocsr()

# recom = model.recommend(userid=0, user_items=visit_csr, N=7)
# print(recom)

#evaluate
actual_list = [[pid] for pid in test['project_id'].values]
predicted_list = []

for uid in test['userCode']:
    recom = model.recommend(userid=users.index(uid), user_items=visit_csr, N=7)
    predicted_list.append([projects[pindex] for pindex, _ in recom])

print('%.10f'%average_precision.mapk(actual_list, predicted_list, k=7))

to_csv = 1
if to_csv:
    test['project_id'] = [' '.join(map(str, pre)) for pre in predicted_list]
    test[['userCode','project_id']].to_csv('submission_{}.csv'.format(model_name), index=False)
# 0.0000319489