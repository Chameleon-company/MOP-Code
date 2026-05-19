from flask import Blueprint, jsonify, render_template, request

from flaskr.logic.home import keyword_search

bp = Blueprint('home', __name__, url_prefix='/')


@bp.route("/", methods=('GET', 'POST'))
def home():
    return render_template('home/index.html', index=True)


@bp.route("/about", methods=('GET', 'POST'))
def about():
    return render_template('home/about.html', about=True)


@bp.route("/faq", methods=('GET', 'POST'))
def faq():
    return render_template('home/faq.html', faq=True)

@bp.route("/contact", methods=('GET', 'POST'))
def contact():
    return render_template('home/contact.html', contact=True)

@bp.route("/data", methods=('GET', 'POST'))
def data():
    return render_template('home/data.html', data=True)
    

@bp.route("/search/datasets")
def datasets():
    search_result = keyword_search(request.args['query'])
    search_result = search_result.sort_values('Downloads', ascending=False)
    return jsonify(search_result.to_dict('records'))
