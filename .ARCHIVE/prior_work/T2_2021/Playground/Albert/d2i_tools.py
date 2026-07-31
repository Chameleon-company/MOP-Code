import numpy as np
import geopandas
import pandas as pd
from shapely.geometry import Polygon, Point
from datetime import datetime, timedelta
from functools import partial
import pyproj
from shapely.ops import transform
from random import sample
from geopy.distance import geodesic

# 001


def geoPolyClip(gdf, polygon):
    """Uses polygon to clip and return geometries inside a gdf.

    Args:
        gdf (geopandas dataframe): pandas dataframe geometry for each row
        polygon (shapely Polygon): to clip and filter the supplied gdf

    Returns:
        filtered geopandas dataframe containing only rows within the supplied polygon
    """
    ngdf = geopandas.clip(gdf, polygon)
    print(f"{ngdf.shape[0]} parking sensors filtered.")
    return ngdf

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
    ngdf = geopandas.clip(gdf, circle)
    print(f"{ngdf.shape[0]} parking sensors filtered.")
    return ngdf

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
        id_col_name (string): name of id column to filter on (e.g. 'bay_id' or 'st_marker_id')
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


# 005
def genPSdata(endTime, fq, period, bayFr, bayTo, dropRate=0):
    """Generate random status for a specified range of parking bays over specified period of time.
    Options are provided to specify frequency (fq) of time interval of status reads,
    and a dropRate to specify a portion of records to be dropped randomly to simulate what we observed in real data.

    Args:
        endTime (date time string): end time of specified period (e.g. "2021-09-02 15:01:23")
        fq (string): specifies time interval between status reads (e.g. "15min")
        period (integer): number of fq time intervals (e.g. for fq="15min", 4*24*7 is for one week's data at 15min intervals)
        bayFr (integer): lower range of specified bay ids
        bayTo (integer): upper range of specified bay ids
        dropRate (int, optional): Defaults to 0. This is portion (between 0 and 1) of data to drop by random sample

    Returns:
        pandas dataframe: of randomly generated status data as specified by the parameters
    """
    w = pd.Series(pd.date_range(end=endTime, periods=period, freq=fq))
    baysList = list(range(bayFr, (bayTo+1)))
    randomisedBaysList = sample(baysList, k=len(baysList))
    lst = []
    for bay in randomisedBaysList:
        a = pd.Series(np.repeat((bay), len(w)))
        aw = pd.concat([a, w], axis='columns')
        aw = aw.sample(frac=(1-dropRate))
        lst.append(aw)

    _df = pd.concat(lst, axis='index')
    _df.columns = ['bay_id', 'db_read_time']

    _df['status'] = np.random.randint(0, 2, size=len(_df))
    _df['status'] = _df['status'].replace({0: "Unoccupied", 1: "Present"})
    _df = _df.sort_values(by=['db_read_time']).reset_index(drop=True)

    startdt = _df['db_read_time'].min()
    enddt = _df['db_read_time'].max()
    print(
        f"Generated {_df.db_read_time.count()} records between {startdt} and {enddt}")
    print(
        f"(with drop rate of {dropRate}, to simulate missing status records in real data).")
    print(
        f"There are {_df.db_read_time.nunique()} time periods over {_df.db_read_time.dt.date.nunique()} days")
    print(f"(frequency of time period = {fq}).")

    return _df


# 006
def getCurrentPSstatus(df):
    """Check supplied df, then print and return occupied and unoccupied parking bay status

    Args:
        df (pandas dataframe): containing full or subset of current status records of parking sensors read

    Returns:
        tuple: of total number of parking bays in df, number of available and unavailable parking bays in df
    """
    no_bays = df.shape[0]
    no_pres = (df["status"] == "Present").sum()
    no_unoc = (df["status"] == "Unoccupied").sum()
    print(
        f"Number of current status records of parking sensors read : {no_bays}")
    print(
        f"Number of current available parking bays : {no_unoc} ({round((100*no_unoc/no_bays),1)}% of total)")
    print(
        f"Number of current unavailable parking bays : {no_pres} ({round((100*no_pres/no_bays),1)}% of total)")
    return no_bays, no_pres, no_unoc


# 007
def timeIntStats(df, startdt, enddt, bin="15min", bin_stat="median", tindex="db_read_time"):
    """Takes a dataframe of periodically read (assume every 15min) status data of parking sensors,
    then filters it based on supplied start and end date times,
    then resamples  this filtered data using supplied bin size for the specified bin statistic.
    The timestamp column name of the input dataframe can be specified (if different from the default).

    Args:
        df (pandas dataframe): of periodically read (assume read at 15min intervals) status data of parking sensors
        startdt (date time string): "yyyy-mm-dd HH:MM:SS" (e.g. "2021-08-28 15:00:00")
        enddt (date time string): "yyyy-mm-dd HH:MM:SS" (e.g. "2021-08-28 15:00:00")
        bin (str, optional): Defaults to "15min". Other appropriate values are "1H", "1D", "1W", "1M"
        bin_stat (str, optional): Defaults to "median". Other optons are "mean", "min", "max", "count"
        tindex (str, optional): Defaults to "db_read_time".

    Returns:
        pandas dataframe: that is a pivot table of the original df, time filtered, and resampled for specified statistics
    """
    df1 = df.copy()
    # in case timestamp column is string --> convert to datetime
    df1[tindex] = pd.to_datetime(df[tindex])
    dfpv = pd.pivot_table(df1, index=tindex, columns="status", aggfunc='count')
    dfpv = dfpv.fillna(0)
    dfpv.columns = ["P", "U"]

    dfpv = dfpv.loc[startdt: enddt]  # filter between specified date times

    if bin_stat == 'median':
        aa = dfpv['P'].resample(bin).median()
        bb = dfpv['U'].resample(bin).median()
    elif bin_stat == 'mean':
        aa = dfpv['P'].resample(bin).mean()
        bb = dfpv['U'].resample(bin).mean()
    elif bin_stat == 'min':
        aa = dfpv['P'].resample(bin).min()
        bb = dfpv['U'].resample(bin).min()
    elif bin_stat == 'max':
        aa = dfpv['P'].resample(bin).max()
        bb = dfpv['U'].resample(bin).max()
    elif bin_stat == 'count':
        aa = dfpv['P'].resample(bin).count()
        bb = dfpv['U'].resample(bin).count()
    else:
        ValueError: print("Invalid 'bin_stat' parameter entered !!")
    # NOTE that I did not include .sum() as we may be resampling
    # (e.g. 4x 15min intervals resample under 1x 60min interval, and it's inappropriate to sum in that case)
    # The .count() included as more of a check the records in a time interval, it shouldn't be used like mean or median

    ndf = pd.concat([aa, bb], axis='columns')
    ndf.columns = ['Present', 'Unoccupied']
    ndf.index.name = 'time_interval'

    # can add below columns to help visualisation labeling (also re-index, but index need to be unique)
    # df['year'] = df.index.year
    # df['qtr'] = df.index.quarter
    # df['mth'] = df.index.month
    # df['wk'] = df.index.week  # week ordinal of the year
    # df['dayOfWk'] = df.index.dayofweek  # day of week, 0=Mon, 6=Sun
    # df['dayOfMth'] = df.index.day  # day of month
    # df['dayOfYr'] = df.index.dayofyear
    print(f"{ndf.shape[0]} time intervals in resampled data")
    return ndf


# 008
# adapted from Miriam's geo filter
def geoCirFilter(gdf, pin, radius):
    """Takes geopandas dataframe with records of lat lon and marker ids, and filters those records within the cicle set by pin and radius

    Args:
        gdf (geopandas dataframe): base list with lat, lon, and marker ids
        pin (tuple of lat lon): pin that is centre of circle centre
        radius (number): radius of circle (in metres)

    Returns:
        [list]: of filtered marker ids that is within the specified circle
    """
    lst_marker_ids = []
    j = 0
    for i in np.arange(0, gdf.shape[0]):
        d = geodesic(pin, (gdf.lati[i], gdf.long[i])).meters
        if d <= radius:
            lst_marker_ids.append(gdf.marker_id[i])
            j = j+1
        else:
            continue
    print(f"{len(lst_marker_ids)} parking sensors filtered.")
    return lst_marker_ids
