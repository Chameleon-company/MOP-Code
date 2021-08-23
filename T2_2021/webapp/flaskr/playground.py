from flaskr.parking_sensor.steps import get_live_parking_json
import functools
import json
import numpy as np
from sodapy import Socrata
import boto3
import pandas as pd
import geopandas as gpd
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, jsonify
)
bp = Blueprint('playground', __name__, url_prefix='/playground')

@bp.route("/", methods=('GET',))
def home():
    return render_template('playground/playground.html')

@bp.route("/traffic_lights", methods=('GET',))
def get_traffic_lights():
    df = pd.read_csv('./data/traffic_lights.csv')
    locations = [[lng, lat] for lng, lat in zip(df['Longitude'], df['Latitude'])]
    return jsonify(locations)

@bp.route("/parking-sensors/now", methods=('GET',))
def get_parking_sensor_latest():
    return get_live_parking_json()

@bp.route("/query_location", methods=('GET',))
def query_location():
    lng = request.args["lng"]
    lat = request.args["lat"]

    # do some location query

    return jsonify((lng,lat))


