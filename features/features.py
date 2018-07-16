
def count_duplicate(data_df, group_by=['userCode']):
    name_col = 'count_'+'_'.join(group_by)
    gp = data_df.groupby(group_by).size().rename(name_col).to_frame().reset_index()
    return gp, name_col
    # target_df = target_df.merge(gp, on='userCode', how='left')
    # return target_df

def prev_visits(data_df, target_df): # todo
    pass

def last_visit(data_df, target_df): # todo
    pass
