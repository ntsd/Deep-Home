import pandas as pd
from scipy import stats

def count_duplicate(data_df, group_cols=['userCode']):
    name_col = 'count_'+'_'.join(group_cols)
    gp = data_df[group_cols][group_cols].groupby(group_cols).size().rename(name_col).to_frame().reset_index()
    return gp, name_col

def prev_visits(data_df, target_df): # todo
    pass

def last_visit(data_df, target_df): # todo
    pass

def item_features_csv():
    # Read cleaned data from previous part using pandas
    project_unit = pd.read_csv('./input/cleaned_project_unit.csv')
    project_main = pd.read_csv('./input/cleaned_project_main.csv')

    # Merge project_unit and project_main together with project id
    project_unit_main = project_main.merge(project_unit, on = ['project_id'])
    # print(project_unit_main.head(3))

    # Create dummie variable for column 'district_id','province_id','project_status','starting_price_range','unit_type_id','amount_bedroom','amount_bathroom','amount_car_parking'
    # dummie_columns = ['district_id','province_id','project_status','starting_price_range','unit_type_id','amount_bedroom','amount_bathroom','amount_car_parking']
    dummie_columns = ['district_id','province_id','unit_type_id']
    for i in dummie_columns:
        dummies = pd.get_dummies(project_unit_main[i], prefix = i)
        project_unit_main = pd.concat([project_unit_main, dummies], axis = 1)
        project_unit_main = project_unit_main.drop(i, axis = 1)

    # binary features
    binary_features = project_unit_main[['project_id','landSize']]
    binary_features = binary_features.groupby(['project_id']).max().reset_index()
    # realed-value features
    con_features = project_unit_main.drop(['landSize'], axis = 1)
    con_features = con_features.groupby(['project_id']).agg(lambda x: stats.mode(x)[0][0]).reset_index()
    item_feature = binary_features.merge(con_features, on = 'project_id')

    # Save project_unit_main dataframe to csv
    item_feature.to_csv('./input/item_feature.csv', index = False)

def user_feature_csv():
    userLog = pd.read_csv('./input/userLog_201801_201802_for_participants.csv', delimiter = ';')

    # Drop unused columns
    userLog = userLog.drop(['year','month','day','hour'], axis = 1)

    # create dummies variable for column 'requestedDevice','userAgent','pageReferrer' and also concat them into userLog (Don't Forget to drop used column)
    dummie_columns = ['requestedDevice','userAgent','pageReferrer']
    for i in dummie_columns:
        dummies = pd.get_dummies(userLog[i], prefix = i)
        userLog = pd.concat([userLog, dummies], axis = 1)
        userLog = userLog.drop(i, axis = 1)

    # Group the userLog by userCode
    userLog = userLog.groupby('userCode').max().reset_index()

    # Drop project_id from userLog
    userLog = userLog.drop('project_id', axis = 1)

    # Save 'userLog' dataframe to csv
    userLog.to_csv('./input/user_feature.csv', index = False)

def create_train_csv(path='./input/userLog_201801_201802_for_participants.csv', delimiter=';'):
    # Read required features using pandas
    train = pd.read_csv(path, delimiter=delimiter) # load userLog
    item_feature = pd.read_csv('./input/item_feature.csv')
    user_feature = pd.read_csv('./input/user_feature.csv')

    # Create new dataframe from userLog using only userCode and project_id column
    # print(userLog.head())
    train = train[['userCode','project_id']]
    # Create new column named num_interact with initial value equals 1
    train['num_interact'] = 1

    # Find total project view for each project by each individual user using group by with userCode and project_id
    train = train.groupby(['userCode','project_id']).sum().reset_index()

    # Merge user_feature to train on userCode
    train = pd.merge(train, user_feature, on='userCode', how='left') 
    # Merge item_feature to train on project_id
    train = pd.merge(train, item_feature, on='project_id', how='left')

    # drop userCode and project_id from train dataframe
    train = train.drop(['userCode','project_id'], axis = 1)

    train.to_csv('./input/train.csv', index = False)

if __name__ == '__main__':
    # item_features_csv()
    # user_feature_csv()
    create_train_csv(path='./input/train_large.csv',delimiter=',')
