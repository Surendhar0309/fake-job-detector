"""Train and compare Logistic Regression, Random Forest and XGBoost on TF-IDF features.

Usage:  python -m src.train --data data/fake_job_postings.csv
"""
import argparse
import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .preprocess import build_text


def main(path: str, out_dir: str = "models", max_features: int = 3000):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(path)
    df["text"] = df.apply(build_text, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["fraudulent"], test_size=0.2, stratify=df["fraudulent"], random_state=42
    )

    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    Xtr, Xte = tfidf.fit_transform(X_train), tfidf.transform(X_test)

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)  # class imbalance
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 scale_pos_weight=pos_weight, eval_metric="logloss", random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        results[name] = {
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
        }
        joblib.dump(model, f"{out_dir}/{name.lower().replace(chr(32), chr(95))}.joblib")

    joblib.dump(tfidf, f"{out_dir}/tfidf.joblib")
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(pd.DataFrame(results).T.to_string())
    best = max(results, key=lambda k: results[k]["Recall"] + results[k]["Precision"])
    with open(f"{out_dir}/best_model.txt", "w") as f:
        f.write(best)
    print("\nBest model (precision+recall):", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fake_job_postings.csv")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    main(a.data, a.out)
