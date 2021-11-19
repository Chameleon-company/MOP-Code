
from flask import Blueprint

bp = Blueprint('utils', __name__, url_prefix='/utils')


@bp.route("/ping", methods=('GET',))
def ping():
    return "alive"
