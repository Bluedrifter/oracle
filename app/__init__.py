from flask import Flask, render_template

from .config import Config
from .core.history import init_db


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    init_db(app.config["DB_PATH"])

    from .modules.hero_villain.routes import bp as hero_villain_bp

    app.register_blueprint(hero_villain_bp)

    @app.route("/")
    def home():
        return render_template("home.html")

    return app
