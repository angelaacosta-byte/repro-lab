"""Regenerate results_summary.csv: the same pipeline run with 5 seeds.

Run:  PYTHONHASHSEED=0 python src/run_seeds.py
      PYTHONHASHSEED=0 python src/run_seeds.py --mlflow   # also log every run to MLflow (optional)

Each row is one seed (5-fold stratified CV). The headline result is the
mean and SD of accuracy_mean ACROSS the 5 seeds, printed at the end.
"""
import argparse, os, subprocess, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import main  # noqa: E402

SEEDS = [0, 1, 2, 3, 42]
OUT = "results_summary.csv"


def code_commit(path="src/train.py"):
    """Short hash of the last commit that changed the training code (stable across later commits)."""
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h", "--", path],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def run_all(seeds=SEEDS):
    commit = code_commit()
    rows = []
    for s in seeds:
        r = main(s)
        rows.append({"params.seed": s,
                     "metrics.accuracy_mean": r["accuracy_mean"],
                     "metrics.accuracy_sd": r["accuracy_sd"],
                     "params.git_commit": commit})
    return pd.DataFrame(rows), r["data_md5"]


def log_to_mlflow(table, data_md5):
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("session5-repro-lab")
    for _, row in table.iterrows():
        with mlflow.start_run(run_name=f"seed-{int(row['params.seed'])}"):
            mlflow.log_params({"model": "logistic_regression", "seed": int(row["params.seed"]),
                               "git_commit": row["params.git_commit"], "data_md5": data_md5})
            mlflow.log_metrics({"accuracy_mean": row["metrics.accuracy_mean"],
                                "accuracy_sd": row["metrics.accuracy_sd"]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlflow", action="store_true", help="also log the runs to MLflow")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    table, data_md5 = run_all()
    table.to_csv(args.out, index=False)
    if args.mlflow:
        log_to_mlflow(table, data_md5)

    acc = table["metrics.accuracy_mean"]
    print(table.to_string(index=False))
    print(f"\nHeadline result (5 seeds): accuracy = {acc.mean():.4f} ± {acc.std(ddof=1):.4f} (mean ± SD)")
    print(f"Range across seeds: {acc.min():.4f} to {acc.max():.4f}  |  data MD5 {data_md5}")
