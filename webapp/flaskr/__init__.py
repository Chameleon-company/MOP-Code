import os

from flask import Flask
from flask_talisman import Talisman


def create_app(test_config=None):
    # Create and configure the Flask application.
    app = Flask(__name__, instance_relative_config=True)

    # Add security-related HTTP headers.
    # HTTPS is disabled for the local Flask development server.
    Talisman(
        app,
        content_security_policy={"default-src": "*"},
        content_security_policy_report_only=True,
        content_security_policy_report_uri="/tools/csp-report",
        force_https=False,
    )

    # Default application configuration.
    app.config.from_mapping(
        SECRET_KEY="dev",
        DATABASE=os.path.join(app.instance_path, "mop.sqlite"),
    )

    # Load instance configuration if it exists.
    if test_config is None:
        app.config.from_pyfile("config.py", silent=True)
    else:
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists.
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Initialise database commands and connection cleanup.
    from . import database

    database.init_app(app)

    # Import application controllers.
    from .controllers import auth
    from .controllers import home
    from .controllers import parking_availability
    from .controllers import tools
    from .controllers import use_cases

    # Register application blueprints.
    app.register_blueprint(auth.bp)
    app.register_blueprint(use_cases.bp)
    app.register_blueprint(tools.bp)
    app.register_blueprint(parking_availability.bp)
    app.register_blueprint(home.bp)

    return app

