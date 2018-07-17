import pandas as pd

def count_duplicate(data_df, group_cols=['userCode']):
    name_col = 'count_'+'_'.join(group_cols)
    gp = data_df[group_cols][group_cols].groupby(group_cols).size().rename(name_col).to_frame().reset_index()
    return gp, name_col

def prev_visits(data_df, target_df): # todo
    pass

def last_visit(data_df, target_df): # todo
    pass

def create_features():
    # Read cleaned data from previous part using pandas
    project_unit = pd.read_csv('./input/cleaned_project_unit.csv')
    project_main = pd.read_csv('./input/cleaned_project_main.csv')

    # Merge project_unit and project_main together with project id
    project_unit_main = project_main.merge(project_unit, on = ['project_id'])
    project_unit_main.head(3)

if __name__ == '__main__':
    create_features()