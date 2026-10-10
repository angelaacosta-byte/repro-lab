"""Automated reproducibility checks. Run from the repository root:  PYTHONHASHSEED=0 pytest -v"""
import hashlib, os, sys
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.train import main          # noqa: E402
from src.run_seeds import run_all   # noqa: E402

EXPECTED_MD5 = "5afc23b9622f8f9ffa60824f4e97d0d5"   # data/breast_cancer.csv
EXPECTED_SEED42 = 0.97367                            # README "Expected output"
EXPECTED_HEADLINE = (0.9786, 0.0031)                 # mean, SD across 5 seeds
TOL = 1e-3                                           # tolerance for floating-point differences


def test_data_hash_matches():
    """Data-version check: the committed data file is exactly the one used for the results."""
    md5 = hashlib.md5(open("data/breast_cancer.csv", "rb").read()).hexdigest()
    assert md5 == EXPECTED_MD5, f"Data file changed: MD5 {md5}"


def test_data_shape_and_target_encoding():
    df = pd.read_csv("data/breast_cancer.csv")
    assert df.shape == (569, 31)
    assert set(df["target"].unique()) == {0, 1}
    assert (df["target"] == 0).sum() == 212   # 0 = malignant
    assert (df["target"] == 1).sum() == 357   # 1 = benign


def test_same_seed_gives_identical_output():
    assert main(42) == main(42)


def test_seed42_matches_expected_accuracy():
    assert main(42)["accuracy_mean"] == pytest.approx(EXPECTED_SEED42, abs=TOL)


def test_results_table_is_reproduced_by_script():
    """results_summary.csv must be regenerated (within tolerance) by src/run_seeds.py."""
    committed = pd.read_csv("results_summary.csv").sort_values("params.seed").reset_index(drop=True)
    fresh, _ = run_all()
    fresh = fresh.sort_values("params.seed").reset_index(drop=True)
    assert list(fresh["params.seed"]) == list(committed["params.seed"])
    for col in ["metrics.accuracy_mean", "metrics.accuracy_sd"]:
        assert fresh[col].to_numpy() == pytest.approx(committed[col].to_numpy(), abs=TOL)
    if fresh["params.git_commit"].iloc[0] != "unknown":           # only when git history is available
        assert (fresh["params.git_commit"] == committed["params.git_commit"].astype(str)).all()


def test_headline_mean_and_sd():
    fresh, _ = run_all()
    acc = fresh["metrics.accuracy_mean"]
    assert acc.mean() == pytest.approx(EXPECTED_HEADLINE[0], abs=TOL)
    assert acc.std(ddof=1) == pytest.approx(EXPECTED_HEADLINE[1], abs=TOL)
