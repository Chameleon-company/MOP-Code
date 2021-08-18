import functools
import json
import pandas as pd
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

@bp.route("/query_location", methods=('GET',))
def query_location():
    lng = request.args["lng"]
    lat = request.args["lat"]

    # do some location query

    return jsonify((lng,lat))
