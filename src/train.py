"""Train and test a model reproducibly.  Run: PYTHONHASHSEED=0 python src/train.py --seed 42"""
import argparse, hashlib, json, random
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def set_seed(seed):
    random.seed(seed)        # Python
    np.random.seed(seed)     # NumPy  (with PyTorch, also: torch.manual_seed(seed))

def main(seed=42, data_path="data/breast_cancer.csv"):
    set_seed(seed)
    df = pd.read_csv(data_path)
    X, y = df.drop(columns="target").values, df["target"].values
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))   # no leakage / sin fuga
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv)
    return {"seed": seed,
            "data_md5": hashlib.md5(open(data_path, "rb").read()).hexdigest(),
            "accuracy_mean": round(float(scores.mean()), 5),
            "accuracy_sd": round(float(scores.std(ddof=1)), 5)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42)
    print(json.dumps(main(ap.parse_args().seed)))
