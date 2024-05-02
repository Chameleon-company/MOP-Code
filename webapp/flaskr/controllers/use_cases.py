from markupsafe import escape
from flask import Blueprint, jsonify, render_template, request, send_from_directory

from flaskr.logic.parking_availability.view_model import build_view_model

bp = Blueprint('use_cases', __name__, url_prefix='/use-cases')

@bp.route("/<name>", methods=('GET',))
def use_cases(name):
    return render_template(f"use-cases/{escape(name)}.html", view_model = get_view_model(name))

@bp.route("/ajax/<name>", methods=('GET',))
def use_cases_ajax(name):
    view_model = get_view_model(name)
    return jsonify(view_model)

"""
    get the view model associated with the use-case name
"""
def get_view_model(name):
    if name is 'parking-availability':
        return build_view_model()
    return {}