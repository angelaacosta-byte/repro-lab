# Reproducible ML Pipeline · Session 5 (UNMSM)

**EN:** Logistic regression on the Wisconsin breast cancer data. Built so a stranger can get the same number.
**ES:** Regresión logística sobre datos de cáncer de mama de Wisconsin. Hecho para que un extraño obtenga el mismo número.

## How to reproduce / Cómo reproducir
```bash
git clone https://github.com/angelaacosta-byte/repro-lab.git && cd repro-lab
pip install -r requirements.txt
PYTHONHASHSEED=0 python src/train.py --seed 42
```

## Expected output / Resultado esperado
```
{"seed": 42, "data_md5": "5afc23b9622f8f9ffa60824f4e97d0d5", "accuracy_mean": 0.97367, "accuracy_sd": 0.01859}
```
Results for 5 seeds / Resultados con 5 semillas: `results_summary.csv`

## Environment / Entorno
Python 3.13.16 · packages in `requirements.txt` · data MD5 `5afc23b9622f8f9ffa60824f4e97d0d5`
