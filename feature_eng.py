import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from utils import (
    pressure_to_hpa, split_city_name, get_lat_lon, get_state,
    split_numeric_categorical_columns, remove_columns_with_nulls,
    remove_records_by_year, impute_rain_columns, cap_outliers_with_iqr,
    remove_records_by_state, impute_missing_values_numerical_columns,
    impute_wind_directions, one_hot_encoding
)

df = pd.read_csv('weatherAUS.csv')

df['Date'] = pd.DatetimeIndex(df['Date'])
years_to_delete = [2007, 2008, 2017]
df = remove_records_by_year(df, 'Date', years_to_delete)

df['Week'] = df['Date'].dt.isocalendar().week


df_location = pd.DataFrame(df['Location'].drop_duplicates())
df_location.reset_index(drop=True, inplace=True)
df_location.sort_values(by='Location', ascending=True, inplace=True)
df_location['Location_split'] = df_location['Location'].apply(split_city_name)
df_location['Latitude'] = None
df_location['Longitude'] = None
df_location['State'] = None

geolocator = Nominatim(user_agent="AUS_geocoding")
df_location[['Latitude', 'Longitude']] = df_location['Location_split'].apply(get_lat_lon).apply(pd.Series)
df_location['State'] = df_location.apply(lambda row: get_state(row['Latitude'], row['Longitude'], row['Location_split']), axis=1)
df = pd.merge(df, df_location, on='Location', how='left')

states_to_keep = ['New South Wales', 'Victoria', 'Queensland', 'Australian Capital Territory']
df = remove_records_by_state(df, 'State', states_to_keep)
df = df.sort_values(by=['Location', 'Date'], axis=0, ascending=[True, True])


df[['Pressure9am', 'Pressure3pm']] = df[['Pressure9am', 'Pressure3pm']].apply(pressure_to_hpa)


df = impute_rain_columns(df)
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1}).astype('object')
df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1}).astype('object')


df = remove_columns_with_nulls(df, df.columns.to_list(), threshold=35)

df.dropna(subset=['RainToday', 'RainTomorrow'], how='any', inplace=True)


numeric_columns, categorical_columns = split_numeric_categorical_columns(df)


column_groups = {
    'Temp_columns': 'Temp',
    'WindSpeed_columns': ['Wind', 'Speed'],
    'Humidity_columns': 'Humidity',
    'Pressure_columns': 'Pressure',
    'Rainfall_columns': 'Rainfall'
}

columns_by_group = {}

for group_name, keywords in column_groups.items():
    columns_by_group[group_name] = []
    for col in numeric_columns:
        if isinstance(keywords, list):
            if all(keyword in col for keyword in keywords):
                columns_by_group[group_name].append(col)
        else:
            if keywords in col:
                columns_by_group[group_name].append(col)

for col_list in columns_by_group.values():
    df = impute_missing_values_numerical_columns(df, col_list)


columns_to_remove = ['Rainfall', 'Latitude', 'Longitude', 'Week']
filtered_columns = []
for col in numeric_columns:
    if col not in columns_to_remove:
        filtered_columns.append(col)

df, outliers_info = cap_outliers_with_iqr(df, filtered_columns)


df['TempRange'] = df['MaxTemp'] - df['MinTemp']
numeric_columns, categorical_columns = split_numeric_categorical_columns(df)


wind_direction_columns = []
for col in df.columns:
    if 'Wind' and 'Dir' in col:
        wind_direction_columns.append(col)

df = impute_wind_directions(df, wind_direction_columns)

df.dropna(subset=['WindGustSpeed', 'WindGustDir'], how='all', inplace=True)

df.dropna(subset=['Pressure3pm', 'Pressure9am'], how='all', inplace=True)


columns_to_remove = ['Week', 'Location_split', 'Latitude', 'Longitude']
filtered_columns = []
for col in df.columns.to_list():
    if col not in columns_to_remove:
        filtered_columns.append(col)

df = df[filtered_columns]


df = one_hot_encoding(df, 'State')
df = one_hot_encoding(df, wind_direction_columns)