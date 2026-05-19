from markupsafe import escape
from flask import Blueprint, jsonify, render_template, request, current_app
from flaskr.logic.parking_availability.view_model import build_view_model
import json

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

# This is the endpoint for Talisman's content security policy (CSP) reports. It currently does nothing
# Since the entire app is destroyed when the AWS lambda function instance is destroyed. However this
# endpoint could easily be replaced by a genuine error logging service in a cloud provider.
@bp.route('/csp-report', methods=['POST'])
def csp_report():
    report = request.get_data()
    return report