from flask.helpers import send_file
from flaskr.parking_sensor.steps import get_live_parking_json, visualize_daily_latest, visualize_hourly_latest
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
    s3_client = boto3.client('s3')

    # get a public link for the parking_sensor.csv
    presigned_url = s3_client.generate_presigned_url('get_object',
        Params={'Bucket': 'opendataplayground.deakin','Key': 'parkingsensor/parkingsensor.csv'},
        ExpiresIn=3600 # 60 minutes
    )

    view_model = {'parking_sensor_collection': presigned_url}

    return render_template('playground/playground.html', view_model = view_model)

@bp.route("/traffic_lights", methods=('GET',))
def get_traffic_lights():
    df = pd.read_csv('./data/traffic_lights.csv')
    locations = [[lng, lat] for lng, lat in zip(df['Longitude'], df['Latitude'])]
    return jsonify(locations)

@bp.route("/parking-sensors/latest.json", methods=('GET',))
def get_parking_sensor_now():
    return get_live_parking_json()

@bp.route("/parking-sensors/daily.png", methods=('GET',))
def get_daily_visualization():
    buffer = visualize_daily_latest()
    return send_file(buffer, mimetype='image/png')

@bp.route("/parking-sensors/hourly.png", methods=('GET',))
def get_hourly_visualization():
    buffer = visualize_hourly_latest()
    return send_file(buffer, mimetype='image/png')

@bp.route("/query_location", methods=('GET',))
def query_location():
    lng = request.args["lng"]
    lat = request.args["lat"]

    # do some location query

    return jsonify((lng,lat))


