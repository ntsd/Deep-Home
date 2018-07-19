import pandas as pd
import numpy as np
import datetime
np.random.seed(44)

def split_data(set_name='', nrows=None):
    user_log_df = pd.read_csv('./input/userLog_201801_201802_for_participants.csv', sep=';', nrows=nrows) 
    # split test to last in user
    # get only duplicate
    user_log_df_dup = user_log_df[user_log_df.duplicated(subset='userCode',keep=False)]
    print('only duplicate user', user_log_df.shape)
    # to get last user duplicate
    df_test = user_log_df_dup.drop_duplicates(subset='userCode', keep="last")[['userCode','project_id']]
    # to drop last deplicate
    df_train = user_log_df[~(user_log_df[['userCode','project_id']].isin(df_test.to_dict('list')).all(axis=1))]
    
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
    # split_data(set_name='small', nrows=100000)
    split_data(set_name='tiny', nrows=10000)
