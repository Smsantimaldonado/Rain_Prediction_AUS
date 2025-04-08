import pandas as pd
import numpy as np
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def pressure_to_hpa(pressure):
    """
    Converts pressure values to a ratio relative to standard atmospheric pressure (hPa).
    Standard atmospheric pressure is approximately 1013.25 hPa ==> 1013.25 hPa = 1 atm.
    
    Parameters:
    - pressure: A numerical value representing pressure (typically in Pascals or similar units).
    Returns:
    - A rounded float representing the pressure as a fraction of standard atmospheric pressure (1013.25 hPa).
    """
    return round(pressure / 1013.25, 5)


def split_city_name(city):
    """
    Splits a city name into two parts if it contains at least two uppercase letters.

    Parameters:
    - city: A string representing the city name.
    Returns:
    - A formatted string where the city name is split into two parts with a space in between, 
      based on the position of the second uppercase letter. 
      If the city name does not contain at least two uppercase letters, it returns the original name.
    """

    matches = []  # List to store the positions of uppercase letters
    iterator = re.finditer(r'[A-Z]', city)  # Find all uppercase letters in the city name
    for match in iterator:
        matches.append(match.start())  # Store the position of each uppercase letter

    if len(matches) >= 2:
        split_index = matches[1]  # Determine the position of the second uppercase letter
        part1 = city[:split_index]  # Extract the first part of the city name
        part2 = city[split_index:]  # Extract the second part of the city name
        return f"{part1} {part2}"  # Return the formatted city name with a space between the two parts
    else:
        return city  # Return the original name if no split is needed


geolocator = Nominatim(user_agent="AUS_geocoding")

def get_lat_lon(city):
    """
    Retrieves the latitude and longitude coordinates of a given city in Australia.

    Parameters:
    - city: A string representing the city name.
    Returns:
    - A tuple (latitude, longitude) if the city is found.
    - (None, None) if the location is not found or an error occurs.
    
    Error Handling:
    - If a `GeocoderTimedOut` or `GeocoderServiceError` occurs, the function retries after a short delay.
    - If another exception occurs, it prints the error message and returns (None, None).
    """

    try: # Attempt to retrieve the city's geolocation (latitude, longitude)
        location = geolocator.geocode(city, exactly_one=True, 
                                      timeout=10, country_codes='au')
        if location: # Return latitude and longitude if a valid location is found
            return location.latitude, location.longitude
        else:
            None, None
    except (GeocoderTimedOut, GeocoderServiceError): # Handle timeout or service errors by retrying after a short delay
        print(f'Geocoder error for {city}. Attempting again..')
        time.sleep(5)
        return get_lat_lon(city)  # Retry the request (recursive call)
    except Exception as e: # Handle any other unexpected exceptions
        print(f"Other error for {city}: {e}")
        return None, None

def get_state(lat, lon, city):
    """
    Retrieves the state or territory name of a given city in Australia based on latitude and longitude.

    Parameters:
    - lat: A float representing the latitude of the city.
    - lon: A float representing the longitude of the city.
    - city: A string representing the city name (used for error messages and retries).
    Returns:
    - A string representing the state or territory name if found.
    - None if the state/territory is not available or an error occurs.

    Error Handling:
    - If a `GeocoderTimedOut` or `GeocoderServiceError` occurs, the function retries after a short delay.
    - If another exception occurs, it prints the error message and returns None.
    """

    try: # Attempt to retrieve the address information for the given coordinates
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
        # If a valid location is found, check if 'state' is in the address data
        if location and 'state' in location.raw['address']:
            return location.raw['address']['state']
        # If 'state' is not found, check if 'territory' is available in the address data
        elif location and 'territory' in location.raw['address']:
            return location.raw['address']['territory']
        else: # Return None if neither state nor territory information is found
            return None
    except (GeocoderTimedOut, GeocoderServiceError):
        # Handle timeout or service errors by retrying after a short delay
        print(f'Geocoder error for {city}. Attempting again..')
        time.sleep(5)
        return get_state(lat, lon, city)  # Recursive call to retry the request
    except Exception as e: # Handle any other unexpected exceptions
        print(f"Other error for {city}: {e}")
        return None


def remove_records_by_year(df, date_column_name, years_to_remove):
    """
    Removes records from a DataFrame based on the years present in a specified date column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_column_name (str): The name of the column containing date values.
        years_to_remove (list): A list of years (integers) whose records should be removed.

    Returns:
        pd.DataFrame: A new DataFrame with records from the specified years removed.
    """
    
    df_to_filter = df.copy()

    # Ensure the specified column is of datetime type
    df_to_filter[date_column_name] = pd.to_datetime(df_to_filter[date_column_name])

    # Create a boolean mask to identify rows to keep
    mask = ~df_to_filter[date_column_name].dt.year.isin(years_to_remove)

    # Filter the DataFrame using the mask
    df_filtered = df_to_filter[mask].copy() # Use .copy() to avoid SettingWithCopyWarning

    return df_filtered


def remove_records_by_state(df, states_column_name, states_to_keep):
    """
    Removes records from a DataFrame based on the years present in a specified date column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_column_name (str): The name of the column containing date values.
        years_to_remove (list): A list of years (integers) whose records should be removed.

    Returns:
        pd.DataFrame: A new DataFrame with records from the specified years removed.
    """
    
    df_to_filter = df.copy()

    # Create a boolean mask to identify rows to keep
    mask = df_to_filter[states_column_name].isin(states_to_keep)

    # Filter the DataFrame using the mask
    df_filtered = df_to_filter[mask].copy() # Use .copy() to avoid SettingWithCopyWarning

    return df_filtered


def split_numeric_categorical_columns(df):
    """
    Splits the columns of a DataFrame into numeric and categorical columns.

    Parameters:
    - df: A pandas DataFrame containing the dataset.
    Returns:
    - numeric_columns: A list of column names that contain numeric data types.
    - categorical_columns: A list of column names that contain categorical data types (e.g., strings or categories).
    """

    # Select and store the column names with numeric data types
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    # Select and store the column names with categorical data types
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    return numeric_columns, categorical_columns


def remove_columns_with_nulls(df, columns_to_check, threshold):
    """
    Removes columns with more than 'threshold' % of missing values.

    Parameters:
    - df: Original DataFrame.
    - columns_to_check: List of columns names that will be checked.
    - threshold: Percentage of missing values allowed before dropping the column.
    Returns:
    - df: Modified DataFrame without the dropped columns.
    - updated_columns_to_check: Updated list of numerical columns.
    """

    # Calculate the percentage of missing values for each column
    nulls_pct = round(df[columns_to_check].isna().sum()*100/df.shape[0],1)

    # Identify columns exceeding the missing value threshold
    columns_to_delete = nulls_pct[nulls_pct > threshold].index.to_list()

    # Drop the identified columns from the DataFrame
    df.drop(columns=columns_to_delete, inplace=True)

    return df


def impute_missing_values_numerical_columns(df, columns):
    """
    Imputes missing values for a given list of numerical columns, considering only data from the same city.
    
    Parameters:
    - df: DataFrame containing the data.
    - imputer: An instance of IterativeImputer (or another imputer).
    - columns: List of numerical columns to be imputed.

    Returns:
    - df: DataFrame with missing values imputed in the specified columns.
    """

    df_imputed = df.copy()

    # Iterate over each city separately
    for city in df_imputed['Location'].unique():
        city_mask = df_imputed['Location'] == city  # Filter rows for the specific city
        city_data = df_imputed.loc[city_mask, columns]  # Select only relevant columns

        # Exclude columns that are completely NaN
        valid_columns = city_data.dropna(axis=1, how="all").columns.tolist()

        if len(valid_columns) > 0:
            imputer = IterativeImputer(max_iter=10, 
                            sample_posterior=True, 
                            n_nearest_features=7, 
                            imputation_order="random", 
                            min_value = city_data.min().min(), 
                            max_value = city_data.max().max(), 
                            random_state=123)
        
            imputed_values = imputer.fit_transform(city_data[valid_columns])
            df_imputed.loc[city_mask, valid_columns] = pd.DataFrame(imputed_values, 
                                                                    index=city_data.index, 
                                                                    columns=valid_columns)

    return df_imputed


def impute_rain_columns(df):
    """
    Imputes missing values in the 'RainToday' and 'RainTomorrow' columns based on related weather data.
    
    Parameters:
    - df: DataFrame containing 'RainToday', 'RainTomorrow', and 'Rainfall' columns.

    Returns:
    - df: DataFrame with missing values imputed in 'RainToday' and 'RainTomorrow'.
    """
    df_imputed = df.copy()

    # Iterate through each unique city
    for city in df_imputed['Location'].unique():
        city_mask = df_imputed['Location'] == city
        city_data = df_imputed.loc[city_mask].copy()

        # Fill 'RainToday' based on 'Rainfall' > 0 (but only where it's NaN)
        missing_today_mask = city_data['RainToday'].isna()  # Mask for missing values
        rainfall_not_na_mask = city_data['Rainfall'].notna()
        city_data.loc[missing_today_mask & rainfall_not_na_mask, 'RainToday'] = (
            city_data.loc[missing_today_mask & rainfall_not_na_mask, 'Rainfall'].gt(0).map({True: 'Yes', False: 'No'})
            )
        
        # If 'Rainfall' is also NaN, check 'RainTomorrow' one row forward to get the previous day's value
        still_missing_mask = city_data['RainToday'].isna()
        city_data.loc[still_missing_mask, 'RainToday'] = (city_data['RainTomorrow'].shift(-1))

        # Now that 'RainToday' is complete, fill 'RainTomorrow' using 'RainToday' of the next day
        city_data['RainTomorrow'] = city_data['RainTomorrow'].fillna(city_data['RainToday'].shift(-1))

        # Update the original DataFrame
        df_imputed.loc[city_mask, ['RainToday', 'RainTomorrow']] = city_data[['RainToday', 'RainTomorrow']]

    return df_imputed


def cap_outliers_with_iqr(df, filtered_columns):
    """
    Caps outliers in the specified columns using the IQR method.
    
    Parameters:
    - df: DataFrame containing the data.
    - filtered_columns: List of numerical columns where outliers should be capped.

    Returns:
    - df: DataFrame with outliers capped in the specified columns.
    """
    
    df_out = df.copy()
    outliers_dict = {}

    for col in filtered_columns:
        q1 = df_out[col].quantile(0.25)
        q3 = df_out[col].quantile(0.75)
        iqr = q3 - q1

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        outliers_dict[col] = {
            'lower_limit': lower_limit,
            'upper_limit': upper_limit
        }

        df_out[col] = np.where(
            df_out[col] > upper_limit, upper_limit,
            np.where(
                df_out[col] < lower_limit, lower_limit,
                df_out[col]
            )
        )

    return df_out, outliers_dict





def group_wind_directions(df, wind_dir_columns):
    """
    Reduces 16 wind direction categories to 8, using angular midpoint logic.

    Parameters:
    - df: pandas DataFrame containing the wind direction columns
    - wind_dir_columns: list of column names to transform

    Returns:
    - df: original DataFrame with additional '_Grouped' columns for each input column
    """
    df_out = df.copy()

    # Dictionary with degrees of the 16 wind directions
    wind_direction_degrees = {
        'N': 0,
        'NNE': 22.5,
        'NE': 45,
        'ENE': 67.5,
        'E': 90,
        'ESE': 112.5,
        'SE': 135,
        'SSE': 157.5,
        'S': 180,
        'SSW': 202.5,
        'SW': 225,
        'WSW': 247.5,
        'W': 270,
        'WNW': 292.5,
        'NW': 315,
        'NNW': 337.5
    }

    # Group for the 8 main directions
    def map_to_8_dirs(direction):
        if pd.isna(direction):
            return np.nan
        deg = wind_direction_degrees.get(direction)
        if deg is None:
            return np.nan
        
        if deg >= 337.5 or deg < 22.5:
            return 'N'
        elif 22.5 <= deg < 67.5:
            return 'NE'
        elif 67.5 <= deg < 112.5:
            return 'E'
        elif 112.5 <= deg < 157.5:
            return 'SE'
        elif 157.5 <= deg < 202.5:
            return 'S'
        elif 202.5 <= deg < 247.5:
            return 'SW'
        elif 247.5 <= deg < 292.5:
            return 'W'
        elif 292.5 <= deg < 337.5:
            return 'NW'
        else:
            return np.nan

    # Mmap each wind direction column to 8 main directions
    for col in wind_dir_columns:
        df_out[col] = df_out[col].map(map_to_8_dirs)

    return df_out









"""
direction_to_degrees = {
    'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
    'S': 180, 'SW': 225, 'W': 270, 'NW': 315
}

degrees_to_direction = {
    (337.5, 360): 'N', (0, 22.5): 'N',
    (22.5, 67.5): 'NE',
    (67.5, 112.5): 'E',
    (112.5, 157.5): 'SE',
    (157.5, 202.5): 'S',
    (202.5, 247.5): 'SW',
    (247.5, 292.5): 'W',
    (292.5, 337.5): 'NW'
}

def degrees_to_cardinal(deg):
    for (low, high), direction in degrees_to_direction.items():
        if low <= deg < high or (low > high and (deg >= low or deg < high)):
            return direction
    return np.nan

def impute_wind_directions(df, columns):
    df_imputed = df.copy()

    # Paso 1: convertir direcciones a grados
    for col in columns:
        df_imputed[col + '_deg'] = df_imputed[col].map(direction_to_degrees)

    # Paso 2: imputar los grados ciudad por ciudad
    deg_columns = [col + '_deg' for col in columns]

    for city in df_imputed['Location'].unique():
        mask = df_imputed['Location'] == city
        city_data = df_imputed.loc[mask, deg_columns]

        # Excluir columnas completamente NaN
        valid_columns = city_data.dropna(axis=1, how='all').columns.tolist()

        if len(valid_columns) > 0:
            imputer = IterativeImputer(max_iter=10, 
                                       random_state=42,
                                       sample_posterior=True,
                                       imputation_order='ascending',
                                       n_nearest_features=5,
                                       min_value=0, 
                                       max_value=360)
            imputed = imputer.fit_transform(city_data[valid_columns])
            df_imputed.loc[mask, valid_columns] = imputed

    # Paso 3: convertir grados nuevamente a direcciones
    for col in columns:
        col_deg = col + '_deg'
        df_imputed[col] = df_imputed[col_deg].apply(lambda x: degrees_to_cardinal(x) if not pd.isna(x) else np.nan)
        df_imputed.drop(columns=[col_deg], inplace=True)

    return df_imputed
"""
