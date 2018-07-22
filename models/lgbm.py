import pandas as pd
import numpy as np

from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import tqdm
import random
random.seed(44)
np.random.seed(44)

import sys
sys.path.append('./')
from features import features
from metrics import average_precision
import matplotlib.pyplot as plt

# Define our recommend_items function with parameters
def recommend_items (userCode, n=7, items_to_ignore=[]):

    # Get userCode's user feature from user_feat
    userCode_feat = user_feat[user_feat['userCode'] == userCode]

    # Create key column for both userCode_feat and item_feat which will be used for merging them together
    userCode_feat['key'] = 1
    item_feat['key'] = 1

    # Merge this userCode's user feature with item feature (item_feat) on key column
    x_userCode = pd.merge(userCode_feat, item_feat, on='key')

    # Drop userCode, project_id and key from x_userCode (no usage in our prediction)
    x_userCode = x_userCode.drop(['userCode','project_id','key'], axis = 1)

    # Make predictions using our decision tree regression model
    predictions = clf.predict(x_userCode)
    # print(predictions)

    # Retrieve indices sorted by num_interact (in predictions) from max to min using argsort
    sorted_indices = np.argsort(predictions)[::-1]

    # Create empty list for appending recommended project_id
    top_n = []

    # Deal with items_to_ignore and append recommended project_id to top_n from sorted indices
    for index in sorted_indices:

        # Get project_id from item_feat using index
        project_id = item_feat.get_value(index, 'project_id')

        # Append this project_id to top_n if it's not in items_to_ignore
        if project_id not in items_to_ignore:
            top_n.append(project_id)

        # Return when top_n size is equal to n
        if len(top_n) >= n:
            return top_n

if __name__ == '__main__':
    is_test = 0
    if is_test:
        train_df = pd.read_csv('./input/train_tiny.csv',delimiter=',')
        test = pd.read_csv('./input/test_tiny.csv')
    else:
        train_df = pd.read_csv('./input/userLog_201801_201802_for_participants.csv',delimiter=';')
        test = pd.read_csv('./input/testing_users.csv',delimiter=';')

    # train_df = train_df[~train_df['project_id'].isin(ignore_project)] # do not for real data

    train, visited_dict = features.create_train(train_df, to_csv=0)

    # Create X_train and y_train from our train data
    from sklearn.model_selection import train_test_split
    train_df_, val_df = train_test_split(train, test_size=0.2)

    x_train = train_df_.drop(['num_interact'], axis = 1)
    y_train = train_df_['num_interact']

    val_train = val_df.drop(['num_interact'], axis = 1)
    val_test = val_df['num_interact']

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(val_train, val_test, reference=lgb_train)
    # Create decision tree regression model
    params = {
        'objective': 'binary',
        'metric': 'logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31, # 31
        'learning_rate': 0.001, # 0.05
        'verbose': 0,
        'nthread': 4,
        'max_depth': -1,
    }

    evals_result = {}

    model = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_train, lgb_eval],
                feature_name=[i for i in train.columns[1:]],
                categorical_feature=[len(train.columns)-1],
                evals_result=evals_result,
                early_stopping_rounds=30,
                verbose_eval=10,
                )

    print("model.best_iteration: ", model.best_iteration)

    predicted_list = []

    # user_feat = pd.read_csv('./input/user_feature.csv')
    # item_feat = pd.read_csv('./input/item_feature.csv') 

    ax = lgb.plot_importance(model, max_num_features=20) 
    plt.savefig('test_importance.png', dpi=600,bbox_inches="tight")

    # # Get userCode's user feature from user_feat
    # userCode_feat = user_feat[user_feat['userCode'].isin(test['userCode'].unique())]
    
    # # Create key column for both userCode_feat and item_feat which will be used for merging them together
    # userCode_feat['key'] = 1
    # item_feat['key'] = 1
    
    # # Merge this userCode's user feature with item feature (item_feat) on key column
    # test_df_feature = pd.merge(userCode_feat, item_feat, on='key')
    
    # del item_feat
    # del userCode_feat

    # y_pred = model.predict(test_df_feature, num_iteration=model.best_iteration)

    # print(y_pred)

    # evaluate = 1
    # if evaluate:#evaluate
    #     actual_list = [[pid] for pid in test['project_id'].values]
    #     print('{:.10f}'.format(average_precision.mapk(actual_list, predicted_list, k=7)))

    
    
