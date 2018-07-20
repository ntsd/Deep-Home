import pandas as pd
import numpy as np
from scipy import stats

def count_duplicate(data_df, group_cols=['userCode']):
    name_col = 'count_'+'_'.join(group_cols)
    gp = data_df[group_cols][group_cols].groupby(group_cols).size().rename(name_col).to_frame().reset_index()
    return gp, name_col

def prev_visits(data_df, target_df): # todo
    pass

def last_visit(data_df, target_df): # todo
    pass

def item_features(to_csv=0):
    # Read cleaned data from previous part using pandas
    project_unit = pd.read_csv('./input/cleaned_project_unit.csv')
    project_main = pd.read_csv('./input/cleaned_project_main.csv')

    # Merge project_unit and project_main together with project id
    project_unit_main = project_main.merge(project_unit, on = ['project_id'])
    # print(project_unit_main.head(3))

    # do quantile data
    quantile_columns = []
    new_cols = []
    for col in quantile_columns:
        i = 'dummy_culumn'
        project_unit_main[i] = np.log(project_unit_main[col])
        q75, q50, q25 = np.percentile(project_unit_main[i].dropna(), [75 ,50, 25])
        min_ = q25
        mid_ = q50
        max_ = q75
        project_unit_main[col+'_range'] = 0
        project_unit_main.loc[project_unit_main[i] < min_, col+'_range'] = 0
        project_unit_main.loc[((project_unit_main[i] > min_) & (project_unit_main[i] < mid_)), col+'_range'] = 1
        project_unit_main.loc[((project_unit_main[i] > mid_) & (project_unit_main[i] < max_)), col+'_range'] = 2
        project_unit_main.loc[project_unit_main[i] > max_, col+'_range'] = 3
        project_unit_main = project_unit_main.drop([col, i], axis = 1)
        new_cols.append(col+'_range')

    # Create dummie variable for column 'district_id','province_id','project_status','starting_price_range','unit_type_id','amount_bedroom','amount_bathroom','amount_car_parking'
    dummie_columns = new_cols + ['district_id','province_id','unit_type_id']
    # dummie_columns = ['district_id','province_id','unit_type_id']
    for i in dummie_columns:
        dummies = pd.get_dummies(project_unit_main[i], prefix = i)
        # if len(dummies.columns)>2: # to create dummy only values len > 2
        project_unit_main = pd.concat([project_unit_main, dummies], axis = 1)
        project_unit_main = project_unit_main.drop(i, axis = 1)

    # change project status A to 1 U to 0
    project_unit_main['project_status'] = project_unit_main['project_status'].apply(lambda x: 1 if x=='A' else 0)

    # add project interacts feature
    userLog = pd.read_csv('./input/userLog_201801_201802_for_participants.csv', delimiter = ';')
    project_interact = userLog.groupby(['project_id']).size().to_frame('project_interact')
    project_unit_main = project_unit_main.merge(project_interact, on='project_id' )

    #norm column
    norm_columns = ['amount_bedroom','amount_bathroom','land_size', 'unit_functional_space_starting_size', 'starting_price', 'project_interact']
    for col in norm_columns:
        project_unit_main[col] = (project_unit_main[col] - project_unit_main[col].min()) / (project_unit_main[col].max() - project_unit_main[col].min())

    # drop unimportance feature
    # userLog = userLog.drop(['userAgent_Other_OS'], axis = 1)
    
    # Save project_unit_main dataframe to csv
    if to_csv:
        project_unit_main.to_csv('./input/item_feature.csv', index = False)

    return project_unit_main

def user_feature(drop_duplicate=0, to_csv=0):
    userLog = pd.read_csv('./input/userLog_201801_201802_for_participants.csv', delimiter = ';')

    # add user interaction
    user_interact = userLog.groupby(['userCode']).size().to_frame('user_interact')
    userLog = userLog.merge(user_interact, on='userCode' )

    # drop duplicate
    if drop_duplicate:userLog = userLog.drop_duplicates(['userCode'],keep='last')
    
    # add week day feature
    import datetime
    userLog['datetime'] = userLog.apply(lambda row : datetime.datetime(row['year'], row['month'], row['day'], row['hour']), axis=1)
    userLog['weekday'] = userLog['datetime'].dt.dayofweek

    # add time_interval feature
    userLog['time_interval'] = '20-23'
    userLog['time_interval'] = np.where((userLog['hour'] >= 0) & (userLog['hour'] < 9), '0-8', userLog['time_interval'])
    userLog['time_interval'] = np.where((userLog['hour'] >= 9) & (userLog['hour'] < 20), '9-19', userLog['time_interval'])

    # Drop unused columns
    userLog = userLog.drop(['year','month','day', 'hour', 'datetime'], axis = 1)

    # create dummies variable for column 'requestedDevice','userAgent','pageReferrer' and also concat them into userLog (Don't Forget to drop used column)
    dummie_columns = ['requestedDevice','userAgent','pageReferrer', 'time_interval', 'weekday']
    for i in dummie_columns:
        dummies = pd.get_dummies(userLog[i], prefix = i)
        userLog = pd.concat([userLog, dummies], axis = 1)
        userLog = userLog.drop(i, axis = 1)

    #norm column
    norm_columns = ['user_interact']
    for col in norm_columns:
        userLog[col] = (userLog[col] - userLog[col].min()) / (userLog[col].max() - userLog[col].min())


    # Group the userLog by userCode
    userLog = userLog.groupby('userCode').max().reset_index()

    # Drop project_id from userLog
    userLog = userLog.drop('project_id', axis = 1)

    # drop unimportance
    # userLog = userLog.drop([], axis = 1)

    # Save 'userLog' dataframe to csv
    if to_csv:
        userLog.to_csv('./input/user_feature.csv', index = False)

    return userLog

def create_train(train=None, to_csv=0):
    # Read required features using pandas
    
    item_feature = pd.read_csv('./input/item_feature.csv')
    user_feature = pd.read_csv('./input/user_feature.csv')

    # Create new dataframe from userLog using only userCode and project_id column
    # print(userLog.head())
    train = train[['userCode','project_id']]

    # Create new column named num_interact with initial value equals 1
    train['num_interact'] = 1

    # Find total project view for each project by each individual user using group by with userCode and project_id
    train = train.groupby(['userCode','project_id']).sum().reset_index()

    # create visited dict
    visited_dict = train.groupby('userCode')['project_id'].apply(lambda x: list(x)).to_dict()

    # Merge user_feature to train on userCode
    train = train.merge(user_feature, on='userCode') 
    # Merge item_feature to train on project_id
    train = train.merge(item_feature, on='project_id')
    # print(train.head())

    # drop userCode and project_id from train dataframe
    train = train.drop(['userCode','project_id'], axis = 1)

    print(train.shape)
    # drop missing value
    train = train.dropna(axis=0)
    print(train.shape)

    if to_csv:
        train.to_csv('./input/train.csv', index = False)
        print('save train.csv success')

    return train, visited_dict

if __name__ == '__main__':
    # item_features(to_csv=1)
    user_feature(drop_duplicate=1, to_csv=1)
    # train = pd.read_csv('./input/userLog_201801_201802_for_participants.csv', delimiter=';') # load userLog
    # create_train(train, to_csv=1)
