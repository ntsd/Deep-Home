import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

import random
random.seed(44)
np.random.seed(44)

import sys
sys.path.append('./')
from features import features
from metrics import average_precision

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
        if len(top_n) == n:
            return top_n

def feature_imprtances(model):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(x_train.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, x_train.columns[indices[f]], importances[indices[f]]))

def ap_func(actual_list, recommend_list, k=7):
    
    m = len(actual_list)
    recoms = []
    precision = 0
    for i, item_ in enumerate(recommend_list):
        if item_ in actual_list:
            recoms.append(1)
            precision += round(sum(recoms[:i+1])/(i+1), 2)
        else:
            recoms.append(0)
          
    ap = round(precision/min(m, k), 2)
    return ap

if __name__ == '__main__':
    # train = pd.read_csv('./input/train.csv')
    train, visited_dict = features.create_train(path='./input/train_small.csv',delimiter=',', to_csv=0)

    # Create X_train and y_train from our train data
    x_train = train.drop(['num_interact'], axis = 1)
    y_train = train['num_interact']

    # Create decision tree regression model
    clf = DecisionTreeRegressor()
    # Fit our train data to the model
    clf.fit(x_train, y_train)
    print('fit model finished')

    feature_imprtances(clf)

    # Read 'user_feature.csv' and 'item_feature.csv' using pandas
    user_feat = pd.read_csv('./input/user_feature.csv')
    item_feat = pd.read_csv('./input/item_feature.csv')

    # Call recommend_items on sample_userCode with 7 recommended projects and no ignored project ids
    # sample_userCode = '00005aba-5ebc-0821-f5a9-bacca40be125'
    # recommend_items(sample_userCode, 7)

    test = pd.read_csv('./input/test_small.csv', nrows=500)

    predicted_list = []

    for uid in test['userCode']:
        # print(visited_dict[uid])
        recom = recommend_items(uid, 7, items_to_ignore=visited_dict[uid]) # todo item ignore
        predicted_list.append(recom)

    evaluate = 1
    if evaluate:#evaluate
        actual_list = [[pid] for pid in test['project_id'].values]
        print('{:.10f}'.format((ap_func(actual_list, predicted_list, k=7))))

    to_csv = 0
    if to_csv:
        test['project_id'] = [' '.join(map(str, pre)) for pre in predicted_list]
        test.to_csv('submission.csv', index=False)
