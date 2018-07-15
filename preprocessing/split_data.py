import pandas as pd
import numpy as np
np.random.seed(44)


if __name__ == '__main__':
    user_log_df = pd.read_csv('../input/userLog_201801_201802_for_participants.csv', sep=';') 
    msk = np.random.rand(len(user_log_df)) < 0.8
    train = user_log_df[msk]
    test = user_log_df[~msk]
    print(len(test))
    print(len(train))
    train.to_csv('../input/train80.csv')
    test.to_csv('../input/test20.csv')
