import functools
from flask import (Blueprint, flash, g, redirect, render_template, request, session, url_for, jsonify)
from werkzeug.security import check_password_hash, generate_password_hash

from flaskr.dataset_search import keyword_search

bp = Blueprint('home', __name__, url_prefix='/')

@bp.route("/", methods=('GET', 'POST'))
def home():
    return render_template('home/index.html')

@bp.route("/about", methods=('GET', 'POST'))
def about():
    return render_template('home/about.html')

@bp.route("/contact", methods=('GET', 'POST'))
def contact():
    return render_template('home/contact.html')

@bp.route("/search/datasets")
def datasets():
    search_result = keyword_search(request.args['query'])
    search_result = search_result.sort_values('Downloads', ascending=False)
    return jsonify(search_result.to_dict('records'))
