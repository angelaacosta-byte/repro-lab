# Reproducible ML Pipeline · Session 5 (UNMSM)

[![Reproducibility check](https://github.com/angelaacosta-byte/repro-lab/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/angelaacosta-byte/repro-lab/actions/workflows/reproducibility.yml)

**EN:** Logistic regression on the Wisconsin Diagnostic Breast Cancer data, built so a stranger can get the same number.
**ES:** Regresión logística sobre los datos de cáncer de mama de Wisconsin, hecha para que un extraño obtenga el mismo número.

Author: Angela Milén Acosta Ticona · Doctoral Program in Deep Tech, AI & Emerging Technologies, UNMSM · Course: Research Methods & Scientific Integrity in AI (Dr. Loveleen Gaur)

## Main result · Resultado principal

| Model | Accuracy, mean ± SD across 5 seeds | Range (worst to best seed) |
|---|---|---|
| Logistic regression (StandardScaler inside a Pipeline, 5-fold stratified CV) | **0.9786 ± 0.0031** | 0.9737 to 0.9824 |

**EN:** The headline result is the mean and standard deviation of the 5-fold cross-validated accuracy across seeds 0, 1, 2, 3 and 42 (`results_summary.csv`). No single seed is reported as the result.
**ES:** El resultado principal es la media y la desviación estándar entre las 5 semillas. Ninguna semilla individual se reporta como resultado.

## How to reproduce · Cómo reproducir

```bash
git clone https://github.com/angelaacosta-byte/repro-lab.git && cd repro-lab
pip install -r requirements-lock.txt          # exact versions of every package (Python 3.13, see .python-version)

PYTHONHASHSEED=0 python src/run_seeds.py      # 1) main result: regenerates results_summary.csv (5 seeds)
PYTHONHASHSEED=0 python src/train.py --seed 42   # 2) single-run check for one seed
PYTHONHASHSEED=0 python src/compare_models.py # 3) own extension: model comparison
pip install pytest==8.4.2 && PYTHONHASHSEED=0 pytest -v   # 4) automated checks
```

Optional: `pip install mlflow` and add `--mlflow` to step 1 to log every run to MLflow (`mlflow.db`, not tracked by Git).

## Expected output · Resultado esperado

Step 1, `src/run_seeds.py`:

```
 params.seed  metrics.accuracy_mean  metrics.accuracy_sd params.git_commit
           0                0.97892              0.01593           f2f2465
           1                0.97893              0.01466           f2f2465
           2                0.97893              0.01466           f2f2465
           3                0.98243              0.00620           f2f2465
          42                0.97367              0.01859           f2f2465

Headline result (5 seeds): accuracy = 0.9786 ± 0.0031 (mean ± SD)
```

Step 2, `src/train.py --seed 42` (one seed only; it is the lowest of the five, so it is a check, not the result):

```
{"seed": 42, "data_md5": "5afc23b9622f8f9ffa60824f4e97d0d5", "accuracy_mean": 0.97367, "accuracy_sd": 0.01859}
```

`params.git_commit` is the last commit that changed `src/train.py`, so every number can be traced to the exact code that produced it.

## Data · Datos

- **Source:** Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B (licence CC BY 4.0). The copy in `data/breast_cancer.csv` was exported from `sklearn.datasets.load_breast_cancer`.
- **Size:** 569 patients, 30 numeric features, 1 target column.
- **Target encoding:** `target = 0` → **malignant** (212 patients); `target = 1` → **benign** (357 patients). This follows scikit-learn's encoding. Note that the positive class (1) is benign, not malignant.
- **Data version:** MD5 `5afc23b9622f8f9ffa60824f4e97d0d5`. The automated test fails if the file changes.

## Environment · Entorno

| File | Purpose |
|---|---|
| `requirements.txt` | Direct dependencies only (numpy, pandas, scikit-learn) |
| `requirements-lock.txt` | **Complete lock file:** every direct and indirect package with its exact version |
| `environment.yml` | Conda alternative that installs the same lock file |
| `.python-version` | Fixes the Python version (3.13) |

The original runs were made in Google Colab with Python 3.13.16. The lock file was resolved in a clean Python 3.13 environment and reproduces the same numbers. GitHub Actions re-checks this on every push.

## Automated check · Verificación automática

`tests/test_reproducibility.py` checks that:

1. the data file has the expected MD5 hash (data-version check) and the expected target encoding;
2. the same seed run twice gives identical output;
3. seed 42 gives the accuracy in "Expected output", within a tolerance of 0.001;
4. `src/run_seeds.py` regenerates `results_summary.csv` within the same tolerance, with the same code commit;
5. the headline mean ± SD is 0.9786 ± 0.0031.

The workflow `.github/workflows/reproducibility.yml` installs `requirements-lock.txt` on a fresh machine and runs all of the above on every push. The badge at the top shows the latest result.

## Own extension · Extensión propia

### 1. Is logistic regression a sensible choice? · Comparación de modelos

`src/compare_models.py` evaluates three models with the same leakage-free protocol (Pipeline, 5-fold stratified CV, 5 seeds). Output: `model_comparison.csv`.

| Model | Accuracy, mean ± SD (5 seeds) | Worst | Best |
|---|---|---|---|
| Majority class (always "benign") | 0.6274 ± 0.0000 | 0.6274 | 0.6274 |
| **Logistic regression** | **0.9786 ± 0.0031** | 0.9737 | 0.9824 |
| Random forest (100 trees, set explicitly) | 0.9600 ± 0.0049 | 0.9543 | 0.9649 |

**EN:** The majority-class baseline shows what "no learning" scores (62.7%). The more complex random forest does **not** beat the simple model, which agrees with Kapoor & Narayanan (2023). The random forest's `n_estimators=100` is written explicitly because the library default changed from 10 to 100 in scikit-learn 0.22.
**ES:** El modelo trivial muestra cuánto se obtiene sin aprender nada (62.7%). El bosque aleatorio, más complejo, **no** supera al modelo simple.

### 2. Reflection: which step is most likely to break reproducibility? · Reflexión

**EN:** In this project the weakest link is the **environment**, not the code or the data. The code is under Git, the data file is committed and fingerprinted, and the seeds are fixed. The first version pinned only three direct packages, so every indirect dependency was whatever pip chose that day. While building the lock file, installing the latest SciPy together with scikit-learn 1.6.1 produced an `OptimizeWarning` from inside the logistic-regression solver. The number did not change this time, but the warning shows that an unpinned dependency can silently change the computation. That is why the full lock file and the CI check were added. A second, quieter risk is the Colab runtime itself: Google updates its Python and packages, so "it ran in Colab" is not a fixed environment.

**ES:** En este proyecto el eslabón más débil es el **entorno**, no el código ni los datos. Al construir el archivo de bloqueo, la versión más reciente de SciPy junto con scikit-learn 1.6.1 generó un aviso dentro del solver de la regresión logística. El número no cambió esta vez, pero muestra que una dependencia sin fijar puede alterar el cálculo sin avisar.

**Link to my thesis · Vínculo con mi tesis:** in my doctoral project (machine learning on Peruvian customs import records, 2022–2026), the riskiest step will instead be the **data**. SUNAT/Veritrade extracts are updated and corrected over time, and the same importer or shipment can appear many times. I will therefore record the MD5 of each extract, as here, and split train and test by importer rather than at random, to avoid the duplicate-patient leakage seen in the Session 5 case study.

## Repository structure · Estructura

```
data/breast_cancer.csv          dataset (committed, MD5-checked)
src/train.py                    one seed: seeded, leakage-free 5-fold CV
src/run_seeds.py                5 seeds → results_summary.csv (+ optional MLflow logging)
src/compare_models.py           own extension → model_comparison.csv
tests/test_reproducibility.py   automated checks (pytest)
.github/workflows/              GitHub Actions: runs the checks on every push
requirements.txt / requirements-lock.txt / environment.yml / .python-version
LICENSE                         MIT (code); the data keep their CC BY 4.0 licence
```

## References · Referencias

- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), 100804. https://doi.org/10.1016/j.patter.2023.100804
- Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

## License · Licencia

Code: MIT License (see `LICENSE`). Data: CC BY 4.0, UCI Machine Learning Repository.
