import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

import tqdm
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
    score_dict = {('_'.join(col.split('_')[:-1]) if col[-1].isdigit() else col):0 for col in x_train.columns.values}
    for f in range(x_train.shape[1]):
        score_dict[('_'.join(x_train.columns[indices[f]].split('_')[:-1]) if x_train.columns[indices[f]][-1].isdigit() else x_train.columns[indices[f]])] += importances[indices[f]]
        print("%d. feature %s (%f)" % (f + 1, x_train.columns[indices[f]], importances[indices[f]]))
    print('--------merge feature importance-------')
    # print(sorted(score_dict.items(), key=lambda kv: kv[1]))
    for feature_name, score in sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True):
        print("feature %s: %f" % (feature_name, score))

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
    is_test = 0
    if is_test:
        train_df = pd.read_csv('./input/train_tiny.csv',delimiter=',')
    else:
        train_df = pd.read_csv('./input/userLog_201801_201802_for_participants.csv',delimiter=';')
    # todo 
    project_count = train_df.groupby(['project_id']).size().to_frame('size')
    size_count = project_count.groupby(['size']).size().to_frame('size_count')
    size_count['size_multiply_count'] = size_count.index * size_count['size_count']
    size_count = size_count.sort_values('size_multiply_count', ascending=False)
    # print(size_count.head())
    print('mean {} std {}'.format(project_count.mean(), project_count.std()))
    min_interacted = float(project_count.mean())# 30 # project_count.mean()
    ignore_project = list(set(project_count[project_count['size'] < min_interacted].index))
    ignore_project = []
    print('ignore_project ', len(ignore_project))
    # train_df = train_df[~train_df['project_id'].isin(ignore_project)] # do not for real data

    train, visited_dict = features.create_train(train_df, to_csv=0)

    # Create X_train and y_train from our train data
    x_train = train.drop(['num_interact'], axis = 1)
    y_train = train['num_interact']

    # Create decision tree regression model
    clf = DecisionTreeRegressor()

    # use random forest for feature selection
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    # clf = RandomForestClassifier()
    # clf = RandomForestRegressor() 

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
    if is_test:
        test = pd.read_csv('./input/test_tiny.csv', nrows=50)
    else:
        test = pd.read_csv('./input/testing_users.csv',delimiter=';')

    predicted_list = []

    with tqdm.tqdm(total=len(test)) as progress: 
        for uid in test['userCode']:
            # print(visited_dict[uid])
            recom = recommend_items(uid, 7, items_to_ignore=visited_dict[uid]+ignore_project) # todo item ignore
            predicted_list.append(recom)
            progress.update(1)

    evaluate = 1
    if evaluate:#evaluate
        actual_list = [[pid] for pid in test['project_id'].values]
        print('{:.10f}'.format(average_precision.mapk(actual_list, predicted_list, k=7)))

    to_csv = 1
    if to_csv:
        test['project_id'] = [' '.join(map(str, pre)) for pre in predicted_list]
        test.to_csv('submission.csv', index=False)
