import numpy as np
import geopandas
import pandas as pd
from shapely.geometry import Polygon, Point
from datetime import datetime, timedelta
from functools import partial
import pyproj
from shapely.ops import transform

# 001


def geoPolyClip(gdf, polygon):
    """Uses polygon to clip and return geometries inside a gdf.

    Args:
        gdf (geopandas dataframe): pandas dataframe geometry for each row
        polygon (shapely Polygon): to clip and filter the supplied gdf

    Returns:
        filtered geopandas dataframe containing only rows within the supplied polygon
    """
    return geopandas.clip(gdf, polygon)

# 002


def geoCirClip(gdf, lat, lon, radius):
    """Uses supplied lat lon of geo point and radius to create a circle buffer,
    to clip and return geometries inside a gdf.

    Args:
        gdf (geopandas dataframe): pandas dataframe geometry for each row
        lat (number): latitude of circle centre
        lon (number): longitude of circle centre
        radius (number): radius in metres of circle

    Returns:
        gdf (geopandas dataframe): filtered gdf containing only rows within the described circle
    """
    circle = cirp(lat, lon, radius)
    return geopandas.clip(gdf, circle)

# 003


def cirp(lat, lon, radius):  # not quite accurate yet
    """Uses lat, lon, and radius (in metres) to create and return a corresponding circle
    (64 sided polygon) buffer.

    Args:
        lat (number): latitude of circle centre
        lon (number): longitude of circle centre
        radius (number): radius in metres of circle

    Returns:
        a shapely Polygon (64 sides) that is roughly the specified circle
    """
    local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(
        lat, lon)

    wgs84_to_aeqd = partial(
        pyproj.transform,
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection))

    aeqd_to_wgs84 = partial(
        pyproj.transform,
        pyproj.Proj(local_azimuthal_projection),
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"))

    pin = Point(lon, lat)
    point_transformed = transform(wgs84_to_aeqd, pin)
    buffer = point_transformed.buffer(radius)

    circle_poly = transform(aeqd_to_wgs84, buffer)
    return circle_poly

# 004


def filterById(ids_list, id_col_name, df):
    """Takes a list of ids, column name of id (in df), and df (dataframe to filter from) and
    returns the filtered df containing only those records with ids from ids_list.

    Args:
        ids_list (list): list of ids to filter
        id_col_name (string): name of id column to filter on
        df (dataframe): pandas dataframe to filter from

    Returns:
        filtered dataframe
    """

    # below some examples and notes on what this function is intended for
    # e.g. id_col_name = "bay_id", and ids_list = [4923, 1253, ..]
    # e.g. id_col_name = "marker_id",  and ids_list = [5921E, 3262S, ..]
    # df can be time data (ETL of parking sensor status)
    #   or base list of parking sensors with lat lon

    return df[df[id_col_name].isin(ids_list)]


# next
# [1] dummy data generator with choice of random distribution
#   return df with some randomly drop records to simulate real data
# [2] time filters
#   keep to a minimum, and make sure it's useful for resampling and visualising time data
