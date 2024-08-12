import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import contextily as ctx
from pyproj import Proj, transform
import pytz
from datetime import datetime, timedelta
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_tide_predictions(station, epoch_time, product='predictions', datum='MLLW', units='metric',
                         time_zone='America/Los_Angeles'):
    base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    local_tz = pytz.timezone(time_zone)
    target_time = datetime.fromtimestamp(epoch_time, local_tz)

    # Fetch predictions for 1 hour before and after the target time
    begin_dt = target_time - timedelta(hours=1)
    end_dt = target_time + timedelta(hours=1)

    begin_utc = begin_dt.astimezone(pytz.UTC)
    end_utc = end_dt.astimezone(pytz.UTC)

    params = {
        'product': product,
        'application': 'NOS.COOPS.TAC.WL',
        'begin_date': begin_utc.strftime('%Y%m%d %H:%M'),
        'end_date': end_utc.strftime('%Y%m%d %H:%M'),
        'datum': datum,
        'station': station,
        'time_zone': 'GMT',
        'units': units,
        'format': 'json'
    }

    response = requests.get(base_url, params=params)

    if response.status_code != 200:
        raise Exception(f"Error fetching data from NOAA: {response.status_code}")

    data = response.json()['predictions']

    df = pd.DataFrame(data)
    df['t'] = pd.to_datetime(df['t'], utc=True)
    df['v'] = df['v'].astype(float)

    df['t'] = df['t'].dt.tz_convert(local_tz)

    # Find the closest prediction to the target time
    df['time_diff'] = abs(df['t'] - target_time)
    closest_prediction = df.loc[df['time_diff'].idxmin()]

    return closest_prediction


def parse_datetime(date_str, time_str):
    try:
        # Try parsing date with 4-digit year
        try:
            date = datetime.strptime(date_str, '%m/%d/%Y')
        except ValueError:
            # If that fails, try with 2-digit year
            date = datetime.strptime(date_str, '%m/%d/%y')

        # Parse time, handling potential leading backtick
        time_str = time_str.replace('`', '')  # Remove backtick if present
        time = datetime.strptime(time_str, '%H-%M-%S').time()

        # Combine date and time
        return datetime.combine(date.date(), time)
    except ValueError as e:
        logging.error(f"Error parsing date/time: {date_str} {time_str}")
        logging.error(f"Error message: {e}")
        return None


def parse_duration(duration_str):
    try:
        # Remove any non-digit characters
        duration_str = ''.join(filter(str.isdigit, duration_str))

        if len(duration_str) != 6:
            raise ValueError(f"Expected 6 digits, got {len(duration_str)}")

        hours = int(duration_str[:2])
        minutes = int(duration_str[2:4])
        seconds = int(duration_str[4:])

        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    except Exception as e:
        logging.error(f"Error parsing duration: {duration_str}")
        logging.error(f"Error message: {e}")
        return None


if __name__ == '__main__':
    root_data_dir = os.path.join('..', '..', 'fieldData', 'referencePole')
    path_df = os.path.join(root_data_dir, 'Summer2024_SunflowerStarReleases_R1.csv')
    df = pd.read_csv(path_df)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['Feature'] = df['Feature'].str.lower()
    print(df.head())

    # Combine date and start time, then calculate midpoint
    df['start_datetime'] = df.apply(lambda row: parse_datetime(row['Date'], row['Start_Time_HH-MM-SS']), axis=1)
    df['duration'] = df['Total_Dive_Time_HH-MM-SS'].apply(parse_duration)

    # Log rows with parsing errors
    error_rows = df[df['start_datetime'].isna() | df['duration'].isna()]
    if not error_rows.empty:
        logging.warning("Rows with parsing errors:")
        for _, row in error_rows.iterrows():
            logging.warning(
                f"Date: {row['Date']}, Start Time: {row['Start_Time_HH-MM-SS']}, Duration: {row['Total_Dive_Time_HH-MM-SS']}")

    # Filter out rows with parsing errors
    df = df.dropna(subset=['start_datetime', 'duration'])

    df['midpoint_time'] = df['start_datetime'] + df['duration'] / 2

    # Convert midpoint time to epoch (Unix timestamp)
    df['midpoint_epoch'] = df['midpoint_time'].astype('int64') / 10 ** 9

    print("\nMidpoint times (first 5 rows):")
    print(df['midpoint_time'].head())
    print("\nMidpoint epoch times (first 5 rows):")
    print(df['midpoint_epoch'].head())

    # Example usage of the modified get_tide_predictions function
    if not df.empty:
        station = '9447130'  # Seattle station, replace with appropriate station
        first_midpoint_epoch = df['midpoint_epoch'].iloc[0]
        closest_prediction = get_tide_predictions(station, first_midpoint_epoch)

        print("\nClosest tide prediction for the first midpoint:")
        print(closest_prediction)
    else:
        logging.error("No valid data after parsing. Please check your input file and data formats.")