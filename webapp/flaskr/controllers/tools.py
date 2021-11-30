from markupsafe import escape
from flask import Blueprint, jsonify, render_template, request

from flaskr.logic.parking_availability.view_model import build_view_model

bp = Blueprint('tools', __name__, url_prefix='/tools')


@bp.route("/<name>", methods=('GET',))
def tools(name):
    return render_template(f"tools/{escape(name)}.html", view_model = get_view_model(name))

"""
    get the view model associated with the use-case name
"""
def get_view_model(name):
    if name is 'parking-availability':
        return build_view_model()
    
    return {}