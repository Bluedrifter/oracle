"""Run with:  python -m app.modules.hero_villain.train"""
import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score

from .model import DATA_PATH, MODEL_PATH, candidates


def train() -> None:
    df = pd.read_csv(DATA_PATH)
    X, y = df["name"], df["alignment"]

    best_name, best_score, best_model = None, -1.0, None
    for name, pipe in candidates().items():
        score = cross_val_score(pipe, X, y, cv=5).mean()
        print(f"{name:22s} accuracy: {score:.3f}")
        if score > best_score:
            best_name, best_score, best_model = name, score, pipe

    best_model.fit(X, y)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nwinner: {best_name} ({best_score:.1%}). the oracle will now be extremely confident about this.")


if __name__ == "__main__":
    train()
