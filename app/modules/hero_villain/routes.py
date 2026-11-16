from flask import Blueprint, current_app, render_template, request

from ...core.confidence import format_confidence
from ...core.history import log_prediction, recent
from .model import predict

bp = Blueprint("hero_villain", __name__, url_prefix="/hero-villain", template_folder="templates")


@bp.route("/", methods=["GET", "POST"])
def index():
    db = current_app.config["DB_PATH"]
    result, error = None, None

    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        if not name:
            error = "the oracle cannot judge a name that does not exist."
        else:
            verdict, prob = predict(name[:50])
            log_prediction(db, "hero_villain", name, verdict, prob)
            result = {"name": name, "verdict": verdict, "conf": format_confidence(prob)}

    return render_template(
        "hero_villain/index.html",
        result=result,
        error=error,
        history=recent(db, "hero_villain"),
    )
