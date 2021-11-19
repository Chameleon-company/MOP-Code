
from flask import Blueprint

bp = Blueprint('util', __name__, url_prefix='/util')


@bp.route("/ping", methods=('GET',))
def ping():
    return "alive"
