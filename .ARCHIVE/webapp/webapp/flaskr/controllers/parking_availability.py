import json
from flask import Blueprint, jsonify, render_template, request
from flask.helpers import send_file

from flaskr.logic.parking_availability.steps import (get_live_parking_json,
                                         visualize_daily_latest,
                                         visualize_filtered_daily_latest,
                                         visualize_filtered_hourly_latest,
                                         visualize_hourly_latest)

bp = Blueprint('parking_availability', __name__, url_prefix='/parking-availability')

@bp.route("/latest.json", methods=('GET',))
def get_parking_sensor_now():
    return get_live_parking_json()


@bp.route("/daily.png", methods=('GET',))
def get_daily_visualization():
    buffer = visualize_daily_latest()
    return send_file(buffer, mimetype='image/png')


@bp.route("/hourly.png", methods=('GET',))
def get_hourly_visualization():
    buffer = visualize_hourly_latest()
    return send_file(buffer, mimetype='image/png')


@bp.route("/daily_filtered.png", methods=('GET',))
def get_filtered_daily_visualization():
    dict = json.loads(request.args['latlng'])
    radius = request.args['radius']

    buffer = visualize_filtered_daily_latest(dict['lat'], dict['lng'], radius)
    return send_file(buffer, mimetype='image/png')


@bp.route("/hourly_filtered.png", methods=('GET',))
def get_filtered_hourly_visualization():
    dict = json.loads(request.args['latlng'])
    radius = request.args['radius']

    buffer = visualize_filtered_hourly_latest(dict['lat'], dict['lng'], radius)
    return send_file(buffer, mimetype='image/png')
