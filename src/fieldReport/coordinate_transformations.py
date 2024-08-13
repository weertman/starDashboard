import os
import numpy as np
import pandas as pd
from get_tides import get_tide_prediction
from pyproj import Proj, CRS, Transformer, Geod
import matplotlib.pyplot as plt
from coordinate_visualization import visualize_survey_points
from datetime import datetime, time
import pytz

def convert_to_datetime64(date_column, start_time_column):
    # Combine Date and Start Time columns
    datetime_str = date_column + ' ' + start_time_column

    # Convert to pandas datetime
    datetime_pd = pd.to_datetime(datetime_str, format='%m/%d/%Y %H-%M-%S')

    # Convert to numpy datetime64
    datetime_np = datetime_pd.to_numpy('datetime64[s]')

    return datetime_np

def calculate_dive_midpoints(start_times, total_dive_times):
    # Preprocess the time strings to replace hyphens with colons
    total_dive_times_processed = total_dive_times.str.replace('-', ':')

    # Convert total dive times to timedelta
    try:
        dive_durations = pd.to_timedelta(total_dive_times_processed)
    except ValueError as e:
        print("Error converting dive times to timedelta. Checking for problematic entries...")
        for idx, time_str in enumerate(total_dive_times_processed):
            try:
                pd.to_timedelta(time_str)
            except ValueError:
                print(f"Invalid time format at index {idx}: '{total_dive_times.iloc[idx]}' (processed as '{time_str}')")
        raise ValueError(
            "Please check the time format in your CSV file and ensure all entries are in 'HH-MM-SS' format.") from e

    # Calculate half of the dive duration
    half_durations = dive_durations / 2

    # Add half of the dive duration to the start times
    midpoint_times = start_times + half_durations.to_numpy()

    return midpoint_times

def get_tidal_height(df, timezone_code='US/Pacific', station_id = '9449880'):
    survey_datetime = convert_to_datetime64(df['Date'], df['Start_Time_HH-MM-SS'])
    survey_midpoints = calculate_dive_midpoints(survey_datetime, df['Total_Dive_Time_HH-MM-SS'])

    tidal_heights = []
    for midpoint in survey_midpoints:
        try:
            tide_info = get_tide_prediction(midpoint, timezone_code, station_id, interpolation_method='cubic')
            tidal_heights.append(tide_info['height'])
        except Exception as e:
            tidal_heights.append(np.nan)
    tidal_heights = np.array(tidal_heights)
    return tidal_heights

def correct_ref_datum_depth(df, timezone_code='US/Pacific', station_id = '9449880'):
    tidal_heights = get_tidal_height(df, timezone_code, station_id)
    depth_ref_datum = df['ref_datum_mllw_depth_M']
    depth_ref_datum = -depth_ref_datum + tidal_heights
    return depth_ref_datum

def get_horizontal_distance_obs(df, timezone_code='US/Pacific', station_id='9449880'):
    depth_ref_datum = correct_ref_datum_depth(df, timezone_code, station_id)
    diff_depth = df['Depth_M'] - np.array(depth_ref_datum)

    # Initialize horizontal_distance_obs with NaN values
    horizontal_distance_obs = np.full(len(df), np.nan)

    # Calculate horizontal distance only for non-direct measurements
    non_direct_measure_idx = ~df['DirectHorizontalMeasure']
    horizontal_distance_obs[non_direct_measure_idx] = np.sqrt(
        (df.loc[non_direct_measure_idx, 'Distance'].values ** 2) -
        (diff_depth[non_direct_measure_idx] ** 2)
    )

    # Set direct horizontal measures
    direct_measure_idx = df['DirectHorizontalMeasure']
    horizontal_distance_obs[direct_measure_idx] = df.loc[direct_measure_idx, 'Distance']

    return horizontal_distance_obs


def normalize_angle(angle):
    """Normalize angle to be between 0 and 360 degrees"""
    return angle % 360

def get_xy_meters_shift_from_ref_pole(df, timezone_code='US/Pacific', station_id='9449880',
                                      magnetic_correction=15.51, compass_flip=180):
    horizontal_distance_obs = get_horizontal_distance_obs(df, timezone_code, station_id)

    # Apply corrections and normalize
    magnetic_heading = df['Degrees'].astype(float)  # Ensure we're working with float values
    corrected_heading = normalize_angle(magnetic_heading + magnetic_correction - compass_flip)

    # Convert to radians
    heading_rad = np.radians(corrected_heading)

    # Calculate shifts
    x_shift = horizontal_distance_obs * np.sin(heading_rad)
    y_shift = horizontal_distance_obs * np.cos(heading_rad)

    return x_shift, y_shift


def xy_to_lat_lon_geodetic(x, y, ref_lat, ref_lon):
    """
    Convert x-y shifts (in meters) to latitude-longitude shifts using geodetic calculations.

    Args:
    x (float or array): X-shift in meters (positive is east)
    y (float or array): Y-shift in meters (positive is north)
    ref_lat (float): Reference latitude in degrees
    ref_lon (float): Reference longitude in degrees

    Returns:
    tuple: (new_lat, new_lon) in degrees
    """
    # Define the EPSG code for WGS84
    wgs84 = CRS("EPSG:4326")

    # Create a projection centered on the reference point
    local_proj = Proj(proj='tmerc', lat_0=ref_lat, lon_0=ref_lon, k=1, x_0=0, y_0=0, ellps='WGS84', units='m')

    # Create a transformer object
    transformer = Transformer.from_proj(local_proj, wgs84, always_xy=True)

    # Project the reference point to the local system
    x0, y0 = local_proj(ref_lon, ref_lat)

    # Add the shifts
    x1, y1 = x0 + x, y0 + y

    # Convert back to lat/lon
    new_lon, new_lat = transformer.transform(x1, y1)

    return new_lat, new_lon


def update_coordinates_geodetic(ref_lat, ref_lon, x_shift, y_shift):
    """
    Update reference coordinates with x-y shifts using geodetic calculations.

    Args:
    ref_lat (float): Reference latitude in degrees
    ref_lon (float): Reference longitude in degrees
    x_shift (float or array): X-shift in meters (positive is east)
    y_shift (float or array): Y-shift in meters (positive is north)

    Returns:
    tuple: (new_lat, new_lon) in degrees
    """
    return xy_to_lat_lon_geodetic(x_shift, y_shift, ref_lat, ref_lon)

def get_lat_long_coords_around_ref_datum (df, timezone_code='US/Pacific', station_id='9449880',
                                       magnetic_correction = 15.51, compass_flip = 180):
    x_shift, y_shift = get_xy_meters_shift_from_ref_pole(df, timezone_code, station_id, magnetic_correction,
                                                         compass_flip)
    new_lat, new_lon = update_coordinates_geodetic(df['ref_datum_lat'], df['ref_datum_long'], x_shift, y_shift)
    return new_lat, new_lon

def get_obs_mllw_depth (df, timezone_code='US/Pacific', station_id='9449880'):
    tidal_heights = get_tidal_height(df, timezone_code, station_id)
    depth_obs = df['Depth_M']

    # Calculate MLLW depth by subtracting tidal height from observed depth
    mllw_depth = depth_obs - tidal_heights

    return mllw_depth

def get_unique_ref_datum_locations (df):
    lats, longs = df['ref_datum_lat'], df['ref_datum_long']
    unique_combos = []
    for lat, long in zip(lats, longs):
        combo = f'{lat}__{long}'
        if combo not in unique_combos:
            unique_combos.append(combo)
    lats, longs = [], []
    for combo in unique_combos:
        combo = combo.split('__')
        lat = float(combo[0])
        long = float(combo[1])
        lats.append(lat)
        longs.append(long)
    return lats, longs

def get_hours_after_earliest_survey(df):
    """
    Takes in the survey dataframe, finds the earliest survey date,
    and creates an array with the same order as df that represents
    hours after the earliest survey date.

    Parameters:
    df (pandas.DataFrame): The survey dataframe containing 'Date' and 'Start_Time_HH-MM-SS' columns.

    Returns:
    numpy.array: An array of float values representing hours after the earliest survey date.
    """
    # Combine 'Date' and 'Start_Time_HH-MM-SS' columns to create a datetime column
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Start_Time_HH-MM-SS'], format='%m/%d/%Y %H-%M-%S')

    # Find the earliest survey date
    earliest_date = df['datetime'].min()

    # Calculate the time difference in hours
    hours_after = (df['datetime'] - earliest_date).dt.total_seconds() / 3600

    return hours_after.values


def get_distance_from_ref_datum(df, timezone_code='US/Pacific', station_id='9449880',
                                magnetic_correction=15.51, compass_flip=180):
    ref_lats, ref_longs = df['ref_datum_lat'], df['ref_datum_long']
    new_lat, new_lon = get_lat_long_coords_around_ref_datum(df, timezone_code, station_id,
                                                            magnetic_correction, compass_flip)

    # Create a Geod object for the WGS84 ellipsoid
    geod = Geod(ellps='WGS84')

    # Calculate distances
    distances = []
    for ref_lat, ref_lon, new_lat_point, new_lon_point in zip(ref_lats, ref_longs, new_lat, new_lon):
        # Calculate the geodesic distance
        _, _, distance = geod.inv(ref_lon, ref_lat, new_lon_point, new_lat_point)
        distances.append(distance)

    return np.array(distances)

def get_night_or_day_binary(df, sunrise='06:00', sunset='21:00',
                            timezone_code='US/Pacific', station_id='9449880'):
    """
    Determine if each survey point was taken during day or night.

    Parameters:
    -----------
    df : pandas.DataFrame
        The survey dataframe containing 'Date' and 'Start_Time_HH-MM-SS' columns.
    sunrise : str, optional
        The sunrise time in 'HH:MM' format. Default is '06:00'.
    sunset : str, optional
        The sunset time in 'HH:MM' format. Default is '21:00'.
    timezone_code : str, optional
        The IANA timezone code for the survey location. Default is 'US/Pacific'.
    station_id : str, optional
        The NOAA station ID code. Not used in this function but kept for consistency.

    Returns:
    --------
    numpy.array
        An array of boolean values where True represents day and False represents night.
    """
    # Convert sunrise and sunset strings to time objects
    sunrise_time = datetime.strptime(sunrise, '%H:%M').time()
    sunset_time = datetime.strptime(sunset, '%H:%M').time()

    # Get the timezone
    tz = pytz.timezone(timezone_code)

    # Combine 'Date' and 'Start_Time_HH-MM-SS' columns to create a datetime column
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Start_Time_HH-MM-SS'], format='%m/%d/%Y %H-%M-%S')

    # Localize the datetime to the specified timezone
    df['datetime'] = df['datetime'].dt.tz_localize(tz)

    # Function to check if a given time is during the day
    def is_daytime(dt):
        return sunrise_time <= dt.time() < sunset_time

    # Apply the function to each datetime in the DataFrame
    day_night_binary = df['datetime'].apply(is_daytime)

    return day_night_binary.values

if __name__ == '__main__':
    path_df = os.path.join('..','..','fieldData', 'referencePole', 'Summer2024_SunflowerStarReleases_R1.csv')
    target_dir = os.path.join('..', '..', 'fieldPlots', 'report1')
    df = pd.read_csv(path_df)
    df = df[df['Feature'] == 'SS']

    survey_datetime = convert_to_datetime64(df['Date'], df['Start_Time_HH-MM-SS'])
    survey_midpoints = calculate_dive_midpoints(survey_datetime, df['Total_Dive_Time_HH-MM-SS'])

    timezone_code = 'US/Pacific'
    station_id = '9449880'  # Friday Harbor station ID
    magnetic_correction = 15.51
    compass_flip = 0

    ref_datum_lats, ref_datum_longs = get_unique_ref_datum_locations(df)
    num_ss = df['Num_SS']
    hours_after_earliest_survey = get_hours_after_earliest_survey(df)
    new_lat, new_lon = get_lat_long_coords_around_ref_datum(df, timezone_code, station_id,
                                                            magnetic_correction, compass_flip)
    mllw_obs_depth = get_obs_mllw_depth(df, timezone_code, station_id)
    distance_from_ref_datum = get_distance_from_ref_datum(df, timezone_code, station_id,
                                                            magnetic_correction, compass_flip)
    day_night_binary = get_night_or_day_binary(df, sunrise='05:30', sunset='20:30')

    obs_num = df['observation_sequence_num']
    degrees = df['Degrees']
    obs_num = [str(n) + ' ' + str(d) for n,d in zip(obs_num, degrees)]

    path_new_df = os.path.join(os.path.dirname(path_df), f'transformed__{os.path.basename(path_df)}')
    df['latitude'] = new_lat
    df['longitude'] = new_lon
    df['MLLW_obs_depth'] = mllw_obs_depth
    df['Hours_after_release'] = hours_after_earliest_survey
    df['Horizontal_distance_from_ref_M'] = distance_from_ref_datum
    df['is_daytime'] = day_night_binary
    df.to_csv(path_new_df, index_label=False)

    x_buffer_scalar, y_buffer_scalar = .8, .5

    try:
        fig, ax = visualize_survey_points(new_lat, new_lon, values=-mllw_obs_depth, labels=obs_num,
                                          output_dir=target_dir,name_fig='site1_depth_map',
                                          title='Circle survey\n2-year old sunflower star survey',
                                          values_label='Observation Depth Below MLLW (m)', cmap='viridis_r',
                                          ref_datum=(ref_datum_lats[0], ref_datum_longs[0]),
                                          x_buffer_scalar=x_buffer_scalar, y_buffer_scalar=y_buffer_scalar, radius_r=15)
        plt.show()

        fig, ax = visualize_survey_points(new_lat, new_lon, values=num_ss,
                                          output_dir=target_dir, name_fig='site1_star_num_map',
                                          title='Circle survey\n2-year old sunflower star survey',
                                          values_label='# of stars observed', cmap='viridis',
                                          ref_datum=(ref_datum_lats[0], ref_datum_longs[0]),
                                          x_buffer_scalar=x_buffer_scalar, y_buffer_scalar=y_buffer_scalar, radius_r=15)
        plt.show()

        fig, ax = visualize_survey_points(new_lat, new_lon, values=hours_after_earliest_survey,
                                          output_dir=target_dir, name_fig='site1_survey_time_map',
                                          title='Circle survey\n2-year old sunflower star survey',
                                          values_label='Hours after release', cmap='viridis',
                                          ref_datum=(ref_datum_lats[0], ref_datum_longs[0]),
                                          x_buffer_scalar=x_buffer_scalar, y_buffer_scalar=y_buffer_scalar, radius_r=15)
        plt.show()

        fig, ax = visualize_survey_points(new_lat, new_lon, values=distance_from_ref_datum,
                                          output_dir=target_dir, name_fig='site1_distance_map',
                                          title='Circle survey\n2-year old sunflower star survey',
                                          values_label='Distance to ref. datum (m)', cmap='viridis',
                                          ref_datum=(ref_datum_lats[0], ref_datum_longs[0]),
                                          x_buffer_scalar=x_buffer_scalar, y_buffer_scalar=y_buffer_scalar, radius_r=15)
        plt.show()

        fig, ax = visualize_survey_points(new_lat, new_lon, values=day_night_binary,
                                          output_dir=target_dir, name_fig='site1_day_night_map',
                                          title='Circle survey\nDay/Night Distribution',
                                          values_label='Day (1) / Night (0)', cmap='coolwarm',
                                          ref_datum=(ref_datum_lats[0], ref_datum_longs[0]),
                                          x_buffer_scalar=x_buffer_scalar, y_buffer_scalar=y_buffer_scalar, radius_r=15)
        plt.show()

    except ValueError as e:
        print(f"Error visualizing survey points: {e}")