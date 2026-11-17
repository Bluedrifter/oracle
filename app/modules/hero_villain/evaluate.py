"""Evaluation harness for the hero/villain module.

Run with:  python -m app.modules.hero_villain.evaluate

With a dataset this small, a single train/test split is mostly noise (a 20% test
set is a dozen names). So every number here comes from repeated stratified
k-fold cross-validation, and the confusion matrix comes from out-of-fold
predictions across the whole dataset.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from .model import DATA_PATH, candidates

DOCS_DIR = Path(__file__).resolve().parents[3] / "docs" / "results"
SEED = 42


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X, y = df["name"], df["alignment"]

    models = {
        # the honest baseline: always guess the most common class
        "baseline_majority": make_pipeline(TfidfVectorizer(), DummyClassifier(strategy="most_frequent")),
        **candidates(),
    }

    repeated = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    results = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X, y, cv=repeated)
        results[name] = {"mean": round(scores.mean(), 4), "std": round(scores.std(), 4)}
        print(f"{name:28s} {scores.mean():.3f} ± {scores.std():.3f}")

    ranked = sorted(
        (k for k in results if k != "baseline_majority"),
        key=lambda k: results[k]["mean"],
        reverse=True,
    )
    best = ranked[0]
    print(f"\nbest: {best} ({results[best]['mean']:.1%})")
    print(f"baseline: {results['baseline_majority']['mean']:.1%}")

    # out-of-fold predictions for the winner -> confusion matrix + report
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = cross_val_predict(models[best], X, y, cv=skf)
    labels = sorted(y.unique())
    print("\n" + classification_report(y, preds, labels=labels))

    cm = confusion_matrix(y, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{best}\n(out-of-fold, n={len(df)})")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "hero_villain_confusion.png", dpi=150)

    payload = {
        "dataset_size": len(df),
        "class_counts": y.value_counts().to_dict(),
        "cv": "5-fold x 5 repeats, stratified, seed 42",
        "models": results,
        "best": best,
    }
    (DOCS_DIR / "hero_villain.json").write_text(json.dumps(payload, indent=2))
    print(f"saved results to {DOCS_DIR}")


if __name__ == "__main__":
    main()
