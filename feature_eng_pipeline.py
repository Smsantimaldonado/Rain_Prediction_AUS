import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder

from utils import (
    pressure_to_hpa, split_city_name, get_lat_lon, get_state,
    split_numeric_categorical_columns, remove_columns_with_nulls,
    remove_records_by_year, impute_rain_columns, remove_records_by_state,
    impute_missing_values_numerical_columns, cap_outliers_with_iqr,
    impute_wind_directions, one_hot_encoding
)


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df.copy()

        df['Date'] = pd.DatetimeIndex(df['Date'])
        # Remover años irrelevantes
        years_to_delete = [2007, 2008, 2017]
        df = remove_records_by_year(df, 'Date', years_to_delete)

        # Crear columna Week (semana del año)
        df['Week'] = df['Date'].dt.isocalendar().week

        # Crear df_location con coordenadas y estado
        df_location = pd.DataFrame(df['Location'].drop_duplicates())
        df_location.reset_index(drop=True, inplace=True)
        df_location.sort_values(by='Location', ascending=True, inplace=True)

        df_location['Location_split'] = df_location['Location'].apply(split_city_name)
        df_location['Latitude'] = None
        df_location['Longitude'] = None
        df_location['State'] = None

        df_location[['Latitude', 'Longitude']] = df_location['Location_split'].apply(get_lat_lon).apply(pd.Series)
        df_location['State'] = df_location.apply(lambda row: get_state(row['Latitude'], row['Longitude'], row['Location_split']), axis=1)

        # Unir info geográfica
        df = pd.merge(df, df_location, on='Location', how='left')

        # Remover registros de islas
        states_to_keep = ['New South Wales', 'Victoria', 'Queensland', 'Australian Capital Territory']
        df = remove_records_by_state(df, 'State', states_to_keep)
        df = df.sort_values(by=['Location', 'Date'], axis=0, ascending=[True, True])

        # Conversión de presión
        df[['Pressure9am', 'Pressure3pm']] = df[['Pressure9am', 'Pressure3pm']].apply(pressure_to_hpa)

        # Imputación de RainToday y RainTomorrow
        df = impute_rain_columns(df)
        df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1}).astype('object')
        df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1}).astype('object')


        df = remove_columns_with_nulls(df, df.columns.to_list(), threshold=35)
        df.dropna(subset=['RainToday', 'RainTomorrow'], how='any', inplace=True)


        # Separar columnas numéricas
        numeric_columns, _ = split_numeric_categorical_columns(df)

        # Crear subgrupos
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

        # Cap de outliers (menos Week, Latitude, Longitude, Rainfall)
        excluded = ['Rainfall', 'Latitude', 'Longitude', 'Week']
        filtered_columns = []
        for col in numeric_columns:
            if col not in excluded:
                filtered_columns.append(col)
        df, _ = cap_outliers_with_iqr(df, filtered_columns)


        # Crear nueva columna TempRange
        df['TempRange'] = df['MaxTemp'] - df['MinTemp']

        numeric_columns, _ = split_numeric_categorical_columns(df)

        # Imputar columnas de viento con transformación a grados
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
        
        # One-hot encoding para State y viento
        df = one_hot_encoding(df, 'State')
        df = one_hot_encoding(df, wind_direction_columns)

        return df