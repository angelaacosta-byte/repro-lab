"""Own extension: is logistic regression actually a good choice?

Compares three models with the SAME leakage-free protocol (Pipeline, 5-fold
stratified CV, 5 seeds):
  1. Majority-class baseline (the "dumb" model: always predict benign)
  2. Logistic regression (the model in train.py)
  3. Random forest with n_estimators=100 written explicitly, so a change in the
     library default (it was 10 before scikit-learn 0.22) cannot change the result.

Run:  PYTHONHASHSEED=0 python src/compare_models.py
Output: model_comparison.csv
"""
import random
import numpy as np, pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2, 3, 42]
DATA = "data/breast_cancer.csv"


def models(seed):
    return {
        "Majority class (baseline)": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "Random forest (100 trees)": make_pipeline(
            StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)),
    }


def compare(seeds=SEEDS, data_path=DATA):
    df = pd.read_csv(data_path)
    X, y = df.drop(columns="target").values, df["target"].values
    per_seed = []
    for s in seeds:
        random.seed(s); np.random.seed(s)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        for name, m in models(s).items():
            per_seed.append({"model": name, "seed": s,
                             "accuracy": float(cross_val_score(m, X, y, cv=cv).mean())})
    per_seed = pd.DataFrame(per_seed)
    summary = (per_seed.groupby("model", sort=False)["accuracy"]
               .agg(accuracy_mean="mean", accuracy_sd=lambda a: a.std(ddof=1),
                    worst="min", best="max", n_seeds="count").round(4).reset_index())
    return summary


if __name__ == "__main__":
    summary = compare()
    summary.to_csv("model_comparison.csv", index=False)
    print(summary.to_string(index=False))
