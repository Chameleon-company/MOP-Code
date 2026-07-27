import functools
import sqlite3

from flask import (
    Blueprint,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from flaskr.database import get_db


bp = Blueprint("auth", __name__, url_prefix="/auth")


@bp.route("/register", methods=("GET", "POST"))
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        error = None

        if not username:
            error = "Username is required."
        elif not email:
            error = "Email is required."
        elif not password:
            error = "Password is required."
        elif len(password) < 8:
            error = "Password must be at least 8 characters."

        if error is None:
            database = get_db()

            try:
                database.execute(
                    """
                    INSERT INTO user (username, email, password)
                    VALUES (?, ?, ?)
                    """,
                    (
                        username,
                        email,
                        generate_password_hash(password),
                    ),
                )
                database.commit()

            except sqlite3.IntegrityError:
                error = "That username or email is already registered."

            else:
                flash(
                    "Registration successful. You can now log in.",
                    "success",
                )
                return redirect(url_for("auth.login"))

        flash(error, "danger")

    return render_template("auth/register.html")


@bp.route("/login", methods=("GET", "POST"))
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        error = None

        if not email:
            error = "Email is required."
        elif not password:
            error = "Password is required."

        user = None

        if error is None:
            user = get_db().execute(
                "SELECT * FROM user WHERE email = ?",
                (email,),
            ).fetchone()

            if user is None:
                error = "Incorrect email or password."
            elif not check_password_hash(user["password"], password):
                error = "Incorrect email or password."

        if error is None:
            session.clear()
            session["user_id"] = user["id"]

            flash("You have successfully logged in.", "success")
            return redirect(url_for("home.home"))

        flash(error, "danger")

    return render_template("auth/login.html")


@bp.before_app_request
def load_logged_in_user():
    user_id = session.get("user_id")

    if user_id is None:
        g.user = None
    else:
        g.user = get_db().execute(
            "SELECT * FROM user WHERE id = ?",
            (user_id,),
        ).fetchone()


@bp.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("home.home"))


def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            flash("Please log in to access that page.", "warning")
            return redirect(url_for("auth.login"))

        return view(**kwargs)

    return wrapped_view
