import pandas as pd
import numpy as np
import datetime
np.random.seed(44)


if __name__ == '__main__':
    user_log_df = pd.read_csv('../input/userLog_201801_201802_for_participants.csv', sep=';') 
    # Create datetime
##    user_log_df['datetime'] = user_log_df.apply(lambda row : datetime.datetime(row['year'], row['month'], row['day'], row['hour']), axis=1)
##    user_log_df['date'] = user_log_df['datetime'].map(lambda x : x.date())
##    
##    date = datetime.date(2018, 2, 20)
##    df_train = user_log_df[user_log_df.date <  date]
##    df_test = user_log_df[user_log_df.date >=  date].sort_values(by = ['userCode', 'datetime'])

    month_ = 2
    day_ = 10
    df_train = user_log_df[(user_log_df.month <=  month_) & (user_log_df.day <  day_)]
    df_test = user_log_df[(user_log_df.month >=  month_) & (user_log_df.day >=  day_)].sort_values(by = ['userCode', 'month', 'day'])
    
    # projects which are in training datasets
    project_train = set(df_train['project_id'].values)
    df_test = df_test[df_test['project_id'].isin(project_train)]
    
    # users which are in training datasets
    user_train = set(df_train['userCode'].values)
    df_test = df_test[df_test['userCode'].isin(user_train)]
    print('df_train',df_train.shape)
    print('df_test', df_test.shape)
    df_train.to_csv('../input/train.csv')
    df_test.to_csv('../input/test.csv')
