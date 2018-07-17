
import pandas as pd
from scipy import stats
import numpy as np

def clean_project_main():
    proj_main = pd.read_csv('./input/project_main.csv', delimiter =';').set_index('project_id') 
    print(proj_main.shape)

    # Display percentage of unique values in each column
    proj_main_size = len(proj_main)
    num_unique = proj_main.nunique()
    # print("Percentage of unique values in each column:\n{}".format(num_unique * 100/proj_main_size))
    proj_main.drop(['created_at', 'lat', 'lon', 'project_name_th',
    'developer_id', 'brand_id','project_name_en'], axis=1, inplace=True)

    # check null and zeros value
    proj_main_size = len(proj_main)
    null_counts = proj_main.isnull().sum()
    # print("Percentage of null values in each column:\n{}".format(null_counts * 100/proj_main_size))
    zeroes_counts = proj_main_size-proj_main.astype(bool).sum()
    # print("Percentage of null values in each column:\n", zeroes_counts * 100/proj_main_size)
    null_zero_counts = (zeroes_counts+null_counts) * 100/proj_main_size
    # print("Percentage of null and zero values in each column:\n", null_zero_counts)
    # drop high percent null
    proj_main.drop(['percent_car_parking', 'highest_price', 'total_unit'], axis=1, inplace=True)

    # Drop rows with NaN columns
    proj_main = proj_main.dropna()

    # All of them must be 0
    # proj_main_size = len(proj_main)
    # null_counts = proj_main.isnull().sum()
    # print("Percentage of null values in each column:\n{}".format(null_counts * 100/proj_main_size)) 

    #land size to one col by convert to wa
    # Sum project land size to have only one column, wa.
    proj_main['landSize'] = proj_main['project_land_size_rai']*400 + proj_main['project_land_size_ngan']*100 + proj_main['project_land_size_wa']
    proj_main = proj_main.drop(['project_land_size_rai','project_land_size_ngan','project_land_size_wa'],axis=1)
    # proj_main['project_land_size_rai'] = proj_main['project_land_size_rai']*400
    # proj_main['project_land_size_ngan'] = proj_main['project_land_size_ngan']*100
    # print(proj_main[['project_land_size_ngan','project_land_size_wa','project_land_size_rai']].head())
    # #get max vault
    # proj_main['landSize'] = proj_main[['project_land_size_ngan','project_land_size_wa','project_land_size_rai']].max(axis=1)

    #change price to 4 category  0 (lowest), 1 (low), 2 (medium), and 3 (high)
    i = 'Log_starting_price'

    # Log Transform
    proj_main[i] = np.log(proj_main['starting_price'])
    q75, q50, q25 = np.percentile(proj_main.Log_starting_price.dropna(), [75 ,50, 25])
    min = q25
    mid = q50
    max = q75
    # plt.figure(figsize=(10,8))
    # plt.subplot(211)
    # plt.xlim(proj_main[i].min(), proj_main[i].max()*1.1)
    # plt.axvline(x=min)
    # plt.axvline(x=mid)
    # plt.axvline(x=max)
    # ax = proj_main[i].plot(kind='kde')
    proj_main['starting_price_range'] = 0
    proj_main.loc[proj_main[i] < min, 'starting_price_range'] = 0
    proj_main.loc[((proj_main[i] > min) & (proj_main[i] < mid)), 'starting_price_range'] = 1
    proj_main.loc[((proj_main[i] > mid) & (proj_main[i] < max)), 'starting_price_range'] = 2
    proj_main.loc[proj_main[i] > max, 'starting_price_range'] = 3
    proj_main = proj_main.drop(['starting_price', i], axis = 1)
    # print(proj_main.head(3))

    #merge faclility
    facility = pd.read_csv('./input/project_facility.csv', delimiter=';')
    # Generate column with 0 for each facility
    proj_main['facility_1'] = 0
    proj_main['facility_2'] = 0
    proj_main['facility_3'] = 0
    proj_main['facility_4'] = 0
    proj_main['facility_5'] = 0
    proj_main['facility_6'] = 0
    # mark as 1 if project has the facility
    for idx,row in facility.iterrows():
        prj = row.project_id
        fac = row.facility_id
        if (fac == 1):
            proj_main.at[prj,'facility_1'] = 1
        if (fac == 2):
            proj_main.at[prj,'facility_2'] = 1
        if (fac == 3):
            proj_main.at[prj,'facility_3'] = 1
        if (fac == 4):
            proj_main.at[prj,'facility_4'] = 1
        if (fac == 5):
            proj_main.at[prj,'facility_5'] = 1
        if (fac == 6):
            proj_main.at[prj,'facility_6'] = 1

    proj_main = proj_main.dropna()
    
    # change project status A to 1 U to 0
    proj_main['project_status'] = proj_main['project_status'].apply(lambda x: 1 if x=='A' else 0)

    # print(proj_main.head(3))
    print(proj_main.shape)
    proj_main.to_csv('./input/cleaned_project_main.csv')

   

def clean_project_unit():
    project_unit = pd.read_csv('./input/project_unit.csv', delimiter = ';')

    # Display percentage of null values in each column
    project_unit_size = len(project_unit)
    null_counts = project_unit.isnull().sum()
    print("Percentage of null values in each column:\n{}".format(null_counts * 100/project_unit_size))

    # Drop the columns with missing values more than 33%
    half_count = len(project_unit) / 3
    project_unit = project_unit.dropna(thresh = half_count, axis = 1)
    # project_unit.head(5)

    # Preparing fill dataframe for method 2
    unit_type_mode = project_unit.drop(['project_id'], axis = 1)
    unit_type_mode = unit_type_mode.groupby(['unit_type_id']).agg(lambda x: stats.mode(x)[0][0]).reset_index()
    # unit_type_mode.head(6)
    # Note: you can use other metrics such as mean, median for this methods.

    # Generate fill key-value by using unit_type_id as a main key
    fill_key_values = unit_type_mode.set_index('unit_type_id').to_dict(orient = 'index')
    print(fill_key_values)

    # Method 1 : Fill missing values with mode grouped by unit_type_id and project_id,
    # Method 2 : Fill missing values with mode grouped by unit_type_id only.

    # Do the Method 1
    project_unit = project_unit.groupby(['project_id','unit_type_id']).agg(lambda x: stats.mode(x)[0][0]).reset_index()
    # project_unit.head(10)

    # do Method 2  Create a new dataframe
    cleaned_project_unit = pd.DataFrame(columns = ['project_id','unit_type_id','amount_bedroom','amount_bathroom','amount_car_parking','unit_functional_space_starting_size'])

    # loop for each row in project_unit, fill the zero value and append it to new dataframe
    for i in project_unit.iterrows():
        unit_type_id = i[1]['unit_type_id']
        #check amount bedroom
        if(i[1]['amount_bedroom'] == 0):
            i[1]['amount_bedroom'] = fill_key_values[unit_type_id]['amount_bedroom']
        #check amount bathroom
        if(i[1]['amount_bathroom'] == 0):
            i[1]['amount_bathroom'] = fill_key_values[unit_type_id]['amount_bathroom']
        #check amount car parking
        if(i[1]['amount_car_parking'] == 0):
            i[1]['amount_car_parking'] = fill_key_values[unit_type_id]['amount_car_parking']
        #check unit functional space starting size
        if(i[1]['unit_functional_space_starting_size'] == 0):
            i[1]['unit_functional_space_starting_size'] = fill_key_values[unit_type_id]['unit_functional_space_starting_size']
        cleaned_project_unit = cleaned_project_unit.append(i[1],ignore_index = True)
    # cleaned_project_unit.head(10)

    # Display percentage of null values in each column
    cleaned_project_unit_size = len(cleaned_project_unit)
    cleaned_null_counts = cleaned_project_unit.isnull().sum()
    print("Percentage of null values in each column:\n{}".format(cleaned_null_counts * 100/cleaned_project_unit_size))

    # Save cleaned project_unit dataframe to csv
    cleaned_project_unit = cleaned_project_unit.set_index('project_id')
    cleaned_project_unit.to_csv('./input/cleaned_project_unit.csv')

if __name__ == '__main__':
    clean_project_main()
    # clean_project_unit()