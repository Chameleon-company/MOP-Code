import os

from flask import Flask
from flask_talisman import Talisman

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    # Talisman is a small Flask extension that handles setting HTTP headers that can help protect
    # against a few common web application security issues (https://github.com/GoogleCloudPlatform/flask-talisman).
    # In particular, this sets the 'x-frame-options=SAMEORIGIN' flag in the HTTP response header to prevent clickjacking.
    # The 'content_security_policy' argument is set to allow content from anywhere or it is too restrictive.
    Talisman(app,
             content_security_policy = {'default-src' : ['\'self\''
             ,'stamen-tiles.a.ssl.fastly.net'
             ,'*.stamen-tiles.a.ssl.fastly.net'
             ,'*.tile.openstreetmap.org'
             ,'*.mapbox.com'
             ,'*.cdnjs.cloudflare.com'
             ]
             ,'style-src' : 
             ['\'self\''
             ,'cdn.jsdelivr.net'
             ,'*.cdn.jsdelivr.net'
             ##,'\'unsafe-inline\''
             ,'maxcdn.bootstrapcdn.com'
             ,'*.maxcdn.bootstrapcdn.com'
             ,'cdnjs.cloudflare.com'
             ,'*.cdnjs.cloudflare.com'
             ,'*.mapbox.com'
             ,'rawcdn.githack.com'
             ]
             ,'script-src': 
             ['\'self\''
             ,'maxcdn.bootstrapcdn.com'
             ,'*.maxcdn.bootstrapcdn.com'
             ,'cdnjs.cloudflare.com'
             ,'*.cdnjs.cloudflare.com'
             ##,'\'unsafe-inline\''
             ,'cdn.jsdelivr.net/npm/'
             ,'*.cdn.jsdelivr.net/npm/'
             ##,'\'unsafe-eval\''
             ,'https://code.jquery.com/jquery-1.12.4.min.js'
             ,'unpkg.com'
             ,'*.unpkg.com'
             ,'*.mapbox.com'
             ,'https://leaflet.github.io/Leaflet.heat/dist/leaflet-heat.js'
             ]
             ,'font-src' : ['\'self\''
             ,'cdnjs.cloudflare.com'
             ,'*.cdnjs.cloudflare.com'
             ,'maxcdn.bootstrapcdn.com'
             ,'*.maxcdn.bootstrapcdn.com'
             ]
             ,'img-src' : ['\'self\''
             ,'cdn.jsdelivr.net'
             ,'*.cdn.jsdelivr.net'
             ,'global.ssl.fastly.net/light_all/'
             ,'*.global.ssl.fastly.net/light_all/'
             ,'*.tile.openstreetmap.org'
             ,'*.cdnjs.cloudflare.com'
             ,'cdnjs.cloudflare.com'
             ,'cartodb-basemaps-b.global.ssl.fastly.net'
             ,'*.cartodb-basemaps-b.global.ssl.fastly.net'
             ,'*.mapbox.com'
             ]
             },
             content_security_policy_report_only = True,
             content_security_policy_report_uri = '/tools/csp-report',
             force_https = True)
    app.config.from_mapping(
        SECRET_KEY='dev'
    )

    from .controllers import use_cases, tools, parking_availability, home
    app.register_blueprint(use_cases.bp)
    app.register_blueprint(tools.bp)
    app.register_blueprint(parking_availability.bp)
    app.register_blueprint(home.bp)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    return app
