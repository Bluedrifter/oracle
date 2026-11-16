from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

DATA_PATH = Path(__file__).parent / "data" / "characters.csv"
MODEL_PATH = Path(__file__).parent / "data" / "model.joblib"

_model = None


def candidates() -> dict:
    """Every contender gets the same character n-grams. Fair fight."""

    def vec():
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)

    return {
        "naive_bayes": make_pipeline(vec(), MultinomialNB()),
        "logistic_regression": make_pipeline(vec(), LogisticRegression(max_iter=1000)),
    }


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            from .train import train

            train()
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(name: str) -> tuple[str, float]:
    model = load_model()
    probs = model.predict_proba([name])[0]
    best = probs.argmax()
    return model.classes_[best], float(probs[best])
