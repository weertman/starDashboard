import requests
import numpy as np
from datetime import datetime, timedelta
import pytz
from scipy import interpolate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_tide_prediction(np_datetime, timezone_code, station_id, interpolation_method='cubic'):
    """
    Fetches the tidal prediction at a specified datetime from a NOAA tidal station
    using interpolation of data points within a time window around the target time.

    Parameters:
    -----------
    np_datetime : numpy.datetime64
        The target date and time for which to retrieve the tidal prediction.
    timezone_code : str
        The IANA timezone code corresponding to the local time of the input datetime.
    station_id : str
        The NOAA station ID code for the tidal station.
    interpolation_method : str, optional
        The type of interpolation to use. Options are 'linear', 'cubic' (default), or 'quadratic'.

    Returns:
    --------
    dict
        A dictionary containing the estimated tidal height.

    Raises:
    -------
    Exception
        If the NOAA API request fails or if no sufficient predictions are found.
    """
    input_datetime = np_datetime.astype('datetime64[s]').astype(datetime)
    tz = pytz.timezone(timezone_code)
    local_datetime = tz.localize(input_datetime)
    utc_datetime = local_datetime.astimezone(pytz.utc)

    start_time = utc_datetime - timedelta(hours=12)
    end_time = utc_datetime + timedelta(hours=12)

    start_str = start_time.strftime('%Y%m%d %H:%M')
    end_str = end_time.strftime('%Y%m%d %H:%M')

    url = (
        f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
        f"begin_date={start_str}&end_date={end_str}&station={station_id}"
        f"&product=predictions&datum=MLLW&units=metric&time_zone=gmt&format=json"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data from NOAA API: {e}")
        raise

    predictions = data.get('predictions', [])
    if not predictions:
        logger.error("No predictions found in the given time window.")
        raise ValueError("No predictions found in the given time window.")

    time_points = []
    height_points = []
    for prediction in predictions:
        pred_time = datetime.strptime(prediction['t'], '%Y-%m-%d %H:%M')
        pred_time = pytz.utc.localize(pred_time)
        time_diff = (pred_time - utc_datetime).total_seconds() / 3600.0  # time in hours
        time_points.append(time_diff)
        height_points.append(float(prediction['v']))

    try:
        f = interpolate.interp1d(time_points, height_points, kind=interpolation_method, fill_value='extrapolate')
        estimated_height = float(f(0))  # 0 represents the target time (midpoint)
    except ValueError as e:
        logger.error(f"Interpolation failed: {e}")
        logger.debug(f"Time points: {time_points}")
        logger.debug(f"Height points: {height_points}")
        raise

    return {
        "datetime": utc_datetime.strftime('%Y-%m-%d %H:%M'),
        "height": estimated_height
    }


if __name__ == '__main__':
    # Example usage
    current_time = np.datetime64(datetime.now())
    timezone_code = 'US/Pacific'
    station_id = '9449880'  # Friday Harbor station ID
    print('current_time:', current_time, 'timezone_code:', timezone_code, 'station_id:', station_id)

    tide_info = get_tide_prediction(current_time, timezone_code, station_id, )
    print('tide info:', tide_info)
