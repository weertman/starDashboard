import os
import numpy as np
from pyproj import Transformer

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import contextily as ctx
from shapely.geometry import Point

from owslib.wms import WebMapService
import io
import warnings
import matplotlib.patheffects as PathEffects

def wgs84_to_web_mercator(lon, lat):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

def visualize_survey_points(latitudes, longitudes, values=None, labels=None, output_dir=None, name_fig=None,
                            point_size=50, title='Underwater Survey Points', cmap='viridis',
                            values_label=None, ref_datum=None, x_buffer_scalar=2, y_buffer_scalar=2,
                            radius_r=None, basemap_source=ctx.providers.Esri.WorldImagery,
                            use_bathymetry=False, figsize=(8,8), label_size=8, label_color='white'):
    """
    Visualize survey points on a map, with optional color coding based on values,
    an optional reference datum point, and an optional circle around the reference datum.

    Parameters:
    latitudes (array-like): List of latitude coordinates
    longitudes (array-like): List of longitude coordinates
    values (array-like, optional): Numeric values for color-coding the points
    labels (list of str, optional): List of strings to display on each coordinate point
    output_dir (str): Directory to save the output image. If None, the image is not saved.
    name_fig (str): Name of the output file (without extension)
    point_size (float): Size of the survey points
    title (str): Title of the plot
    cmap (str): Colormap to use for the points when values are provided
    values_label (str): Label for the colorbar
    ref_datum (tuple, optional): (latitude, longitude) of the reference datum point
    x_buffer_scalar (float): Factor by which the x-axis is padded based on longitude range
    y_buffer_scalar (float): Factor by which the y-axis is padded based on latitude range
    radius_r (float, optional): Radius in meters for a circle around the reference datum point
    basemap_source (contextily.tile_provider): Source for the basemap. Default is Esri World Imagery.
    use_bathymetry (bool): If True, use NOAA's nautical charts with bathymetric data
    figsize (tuple): Figure size in inches
    label_size (int): Font size for the labels
    label_color (str): Color of the label text

    Returns:
    fig, ax: The matplotlib figure and axis objects
    """
    # Convert inputs to numpy arrays and remove any NaN or Inf values
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)
    valid_indices = np.isfinite(latitudes) & np.isfinite(longitudes)

    if values is not None:
        values = np.array(values)
        valid_indices &= np.isfinite(values)

    latitudes = latitudes[valid_indices]
    longitudes = longitudes[valid_indices]
    if values is not None:
        values = values[valid_indices]
    if labels is not None:
        labels = np.array(labels)[valid_indices]

    if len(latitudes) == 0 or len(longitudes) == 0:
        raise ValueError("No valid coordinates found after removing NaN and Inf values")

    # Convert coordinates to Web Mercator
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_coords, y_coords = transformer.transform(longitudes, latitudes)

    # Create a GeoDataFrame with the converted points
    data = {'geometry': [Point(x, y) for x, y in zip(x_coords, y_coords)]}
    if values is not None:
        data['values'] = values
    survey_points = gpd.GeoDataFrame(data, crs="EPSG:3857")

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the survey points
    if values is not None:
        if values_label is None:
            values_label = 'Values'
        scatter = ax.scatter(x_coords, y_coords, c=values, s=point_size, cmap=cmap, label='Survey Points')
        plt.colorbar(scatter, ax=ax, label=values_label, shrink=0.3)
    else:
        scatter = ax.scatter(x_coords, y_coords, color='red', s=point_size, label='Survey Points')

    # Add labels if provided
    if labels is not None:
        if len(labels) != len(x_coords):
            raise ValueError("Number of labels must match number of coordinates")
        for x, y, label in zip(x_coords, y_coords, labels):
            text = ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=label_size, color=label_color, weight='bold')
            text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

    # Plot the reference datum point and circle if provided
    if ref_datum is not None:
        ref_lat, ref_lon = ref_datum
        ref_x, ref_y = transformer.transform(ref_lon, ref_lat)
        ax.scatter(ref_x, ref_y, color='yellow', s=point_size * 5, marker='*',
                   edgecolor='black', linewidth=1, label='Reference Datum')

        # Add circle if radius is provided
        if radius_r is not None:
            # Convert radius from meters to Web Mercator units
            # Note: This is an approximation and works best for small radii
            ref_lat_rad = np.radians(ref_lat)
            radius_mercator = radius_r / np.cos(ref_lat_rad)

            circle = Circle((ref_x, ref_y), radius_mercator, fill=False,
                            edgecolor='yellow', linestyle='--', linewidth=2)
            ax.add_patch(circle)

    # Set the extent of the map
    bounds = survey_points.total_bounds
    x_buffer = (bounds[2] - bounds[0]) * x_buffer_scalar
    y_buffer = (bounds[3] - bounds[1]) * y_buffer_scalar
    ax.set_xlim(bounds[0] - x_buffer, bounds[2] + x_buffer)
    ax.set_ylim(bounds[1] - y_buffer, bounds[3] + y_buffer)

    # Add the basemap
    if use_bathymetry:
        try:
            # NOAA Nautical Charts (includes bathymetry)
            wms_url = 'https://gis.charttools.noaa.gov/arcgis/services/MCS/ENCOnline/MapServer/WMSServer'
            wms = WebMapService(wms_url)

            # Get the bounding box in the correct format
            bbox = (bounds[0], bounds[1], bounds[2], bounds[3])

            # Request the image
            img = wms.getmap(layers=['0'], srs='EPSG:3857', bbox=bbox, size=(1000, 1000), format='image/png',
                             transparent=True)

            # Convert the image to a numpy array
            img_array = plt.imread(io.BytesIO(img.read()))

            # Add the image to the plot
            ax.imshow(img_array, extent=bbox, alpha=0.7)
        except Exception as e:
            warnings.warn(f"Failed to load bathymetry data: {str(e)}. Falling back to default basemap.")
            ctx.add_basemap(ax, source=basemap_source, zoom=19, attribution='')
    else:
        ctx.add_basemap(ax, source=basemap_source, zoom=19, attribution='')

    plt.title(title, loc='left')
    plt.legend()
    plt.axis('off')  # Turn off axis labels

    plt.tight_layout()

    # Save the plot if output_dir is provided
    if output_dir:
        if name_fig is None:
            name_fig = 'survey_points_map'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{name_fig}.png'), dpi=300, bbox_inches='tight')

    return fig, ax