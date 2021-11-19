from flask import Blueprint, jsonify, render_template, request

from flaskr.dataset_search import keyword_search

bp = Blueprint('home', __name__, url_prefix='/')


@bp.route("/", methods=('GET', 'POST'))
def home():
    return render_template('home/index.html')


@bp.route("/search/datasets")
def datasets():
    search_result = keyword_search(request.args['query'])
    search_result = search_result.sort_values('Downloads', ascending=False)
    return jsonify(search_result.to_dict('records'))
