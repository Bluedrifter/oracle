from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

DATA_PATH = Path(__file__).parent / "data" / "characters.csv"
MODEL_PATH = Path(__file__).parent / "data" / "model.joblib"

_model = None


def candidates() -> dict:
    """Every contender gets a fair fight on the same data."""

    def char_vec(lo=2, hi=4):
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(lo, hi), lowercase=True)

    def word_vec():
        return TfidfVectorizer(analyzer="word", lowercase=True)

    return {
        "naive_bayes_char_2_4": make_pipeline(char_vec(2, 4), MultinomialNB()),
        "naive_bayes_char_2_5": make_pipeline(char_vec(2, 5), MultinomialNB()),
        "logreg_char_2_4": make_pipeline(char_vec(2, 4), LogisticRegression(max_iter=1000)),
        "logreg_word": make_pipeline(word_vec(), LogisticRegression(max_iter=1000)),
        "random_forest_char_2_4": make_pipeline(
            char_vec(2, 4), RandomForestClassifier(n_estimators=200, random_state=42)
        ),
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
