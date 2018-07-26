import pandas as pd
import numpy as np
import datetime
np.random.seed(44)

def split_data(set_name='', nrows=None):
    user_log_df = pd.read_csv('./input/userLog_201801_201802_for_participants.csv', sep=';', nrows=nrows) 
    # split test to last in user
    # get only duplicate
    user_log_df_dup = user_log_df[user_log_df.duplicated(subset='userCode',keep=False)]
    print('only duplicate user', user_log_df_dup.shape)
    # to get last user duplicate
    df_test = user_log_df_dup.drop_duplicates(subset='userCode', keep="last")
    # print(df_test.head())
    # to drop last deplicate
    df_train = user_log_df[~(user_log_df.index.isin(df_test.index))]

    print('df_train',df_train.shape)

    # drop duplicate projectid usercode to test
    userproject = [row['userCode']+str(row['project_id']) for index, row in df_test.iterrows()]
    df_train['userproject'] = [row['userCode']+str(row['project_id']) for index, row in df_train.iterrows()]
    df_train = df_train[~df_train['userproject'].isin(userproject)]

    # print(df_train.head(10))
    print('df_train',df_train.shape)
    print('df_test', df_test.shape)

    # delete test that userCode and project_id is in train
    # df_test = df_test.merge(df_train,on=['userCode','project_id'], how='left', indicator=True).dropna()

    # projects which are in training datasets
    project_train = set(df_train['project_id'].values)
    df_test = df_test[df_test['project_id'].isin(project_train)]
    
    # users which are in training datasets
    user_train = set(df_train['userCode'].values)
    df_test = df_test[df_test['userCode'].isin(user_train)]

    print('df_train',df_train.shape)
    print('df_test', df_test.shape)
    df_train.to_csv('./input/train_{}.csv'.format(set_name))
    df_test.to_csv('./input/test_{}.csv'.format(set_name))

if __name__ == '__main__':
    split_data(set_name='10000', nrows=10000)
    # split_data(set_name='medium', nrows=50000)
    # split_data(set_name='big', nrows=100000)
    
