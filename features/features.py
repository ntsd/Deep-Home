
def count_duplicate(data_df, group_cols=['userCode']):
    name_col = 'count_'+'_'.join(group_cols)
    gp = data_df[group_cols][group_cols].groupby(group_cols).size().rename(name_col).to_frame().reset_index()
    # print(gp.head())
    return gp, name_col
    # target_df = target_df.merge(gp, on='userCode', how='left')
    # return target_df

def prev_visits(data_df, target_df): # todo
    pass

def last_visit(data_df, target_df): # todo
    pass
