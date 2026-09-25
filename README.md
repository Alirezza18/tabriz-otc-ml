# TabrizOTC-ML — Morphology-Driven Outdoor Thermal Comfort Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-28%20passing-brightgreen)](#reproducibility)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs00484--025--02921--8-blue)](https://doi.org/10.1007/s00484-025-02921-8)

Reference implementation of the machine-learning pipeline from:

> **Alinasab, N., Mohammadzadeh, N., Karimi, A., Mohammadzadeh, R., & Gál, T. (2025).**
> *A measurement-based framework integrating machine learning and morphological dynamics for outdoor thermal regulation.*
> International Journal of Biometeorology, 69(9), 1645–1662.
> [https://doi.org/10.1007/s00484-025-02921-8](https://doi.org/10.1007/s00484-025-02921-8) (open access, CC-BY 4.0)

The published study combines a 62-day field campaign in Tabriz, Iran (1,168 measurement points across
173 street canyons and 7 neighborhoods) with six Bayesian-optimized machine-learning classifiers to
predict three outdoor thermal comfort (OTC) indices — **UTCI**, **PET** and **PMV** — from street
morphology, and then explains the models with **SHAP** value analysis.

This repository reproduces that workflow end to end, from the physical measurement equations to the
SHAP interpretation stage.

---

## What the pipeline does

```
                    ┌─────────────────────────────────────────────────────┐
                    │  Street-morphology features        Microclimate     │
                    │  ─ orientation (45° bins)          ─ Ta, RH, V, Tg  │
                    │  ─ street width (2–38 m)                            │
                    │  ─ H/W ratio (0.1–3)                                │
                    │  ─ building floors (1–8), trees                     │
                    └───────────────────────┬─────────────────────────────┘
                                            │
              Eq. 1 (globe thermometer Tmrt) │  Eq. 2 (log wind profile)
                                            ▼
                    ┌─────────────────────────────────────────────────────┐
                    │  OTC indices: UTCI / PET / PMV                      │
                    │  → 3 thermal-stress classes each (paper Table 2)    │
                    └───────────────────────┬─────────────────────────────┘
                                            │  70/30 split, balanced
                                            │  class weights
                                            ▼
                    ┌─────────────────────────────────────────────────────┐
                    │  6 classifiers: KNN, DT, SVM, RF, XGBoost, CatBoost │
                    │  hyperparameters tuned per model with GP-EI         │
                    │  Bayesian optimization (5-fold CV objective)        │
                    └───────────────────────┬─────────────────────────────┘
                                            ▼
                    ┌─────────────────────────────────────────────────────┐
                    │  Weighted accuracy / precision / recall / F1        │
                    │  (Table 3)  +  SHAP feature attribution (Figs. 3–4) │
                    └─────────────────────────────────────────────────────┘
```

### Faithfulness to the paper

| Paper element | Implementation |
|---|---|
| Globe-thermometer mean radiant temperature (Eq. 1) | `otc_ml.data.mean_radiant_temp` — D = 150 mm, ε = 0.95 |
| Logarithmic wind-speed profile (Eq. 2) | `otc_ml.data.wind_at_height` — z₀ = 0.01 m (urban) |
| Table 1 morphological ranges | Enforced + validated in `otc_ml.features` |
| Table 2 three-class definitions | `otc_ml.config.OTC_CLASS_DEFS` (identical boundaries) |
| 70/30 train–test split | `otc_ml.pipeline.run_single_index` (stratified) |
| Class imbalance via balanced sample weights | `otc_ml.models.balanced_sample_weights` (`compute_sample_weight(class_weight="balanced")`) |
| GP surrogate + Expected Improvement BO, 5-fold CV objective, log-uniform learning rates | `otc_ml.bayes_opt` (Optuna `GPSampler`, EI acquisition) |
| Six algorithms | `otc_ml.models.ALGORITHMS` |
| Weighted-average metrics (Table 3) | `otc_ml.evaluate` |
| SHAP per-class importance (Fig. 3) and binary direction analysis (Fig. 4) | `otc_ml.explain` |

The paper's headline results (best model per index) ship as `otc_ml.config.BEST_MODELS` /
`EXPECTED_RESULTS` so any run can be compared against Table 3 directly.

---

## Installation

```bash
git clone https://github.com/Alirezza18/tabriz-otc-ml.git
cd tabriz-otc-ml
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

Python 3.10+ recommended. `requirements.txt` pins the packages the study needs:
`numpy`, `pandas`, `scikit-learn`, `optuna`, `xgboost`, `catboost`, `shap`.

## Quickstart

```bash
# Full study on the synthetic benchmark (60 BO trials × 6 algorithms × 3 indices;
# takes a while — roughly the paper's complete configuration):
python scripts/run_study.py --out results/

# Fast demonstration run:
python scripts/run_study.py --n-trials 5 --out results/

# Use your own field-measurement CSV (column layout below):
python scripts/run_study.py --csv data/my_measurements.csv --out results/
```

Outputs written under `results/`:

| File | Content |
|---|---|
| `results_<UTCI|PET|PMV>.csv` | one Table-3-style metrics row per algorithm |
| `best_params_<index>.json` | winning hyperparameters per algorithm |
| `shap_importance_<index>.csv` | per-class mean \|SHAP\| per feature (Fig. 3 equivalent) |
| `shap_directions_<index>.csv` | binary high/low-value class-direction analysis (Fig. 4 equivalent) |
| `summary.json` | best model per index + all metrics |

### Run with Docker (no Python setup needed)

The study runs headless in a container and writes its outputs to your host:

```bash
# Full study (60 BO trials x 6 algorithms x 3 indices) - results in ./results:
docker run --rm -v "$PWD/results:/app/results" ghcr.io/alirezza18/tabriz-otc-ml:latest

# Fast demo:
docker run --rm -v "$PWD/results:/app/results" ghcr.io/alirezza18/tabriz-otc-ml:latest \
    --n-trials 3 --out results/

# With your own field-measurement CSV (drop it into ./data first):
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
    ghcr.io/alirezza18/tabriz-otc-ml:latest --csv data/my_measurements.csv --out results/
```

Or with compose: `docker compose up` (edit the `command:` line for demo/CSV modes).
The image runs as an unprivileged user and is rebuilt + smoke-tested (a real
mini-study inside the container) on every change; every `vX.Y.Z` release tag
publishes a matching image version.

### Using your own dataset

Bring a CSV with these columns (the synthetic benchmark generates exactly this layout):

```
orientation_deg, street_width_m, hw_ratio, n_floors, n_trees, Ta, RH, V, Tmrt, UTCI, PET, PMV
```

`orientation_deg` is the street axis in degrees (0–180). If `Tmrt` is absent it is derived from
globe temperature; pass `Ta, RH, V` and the index columns you have — see
`otc_ml.data.load_dataset` for the exact contract.

### Library usage

```python
from otc_ml.data import build_dataset, mean_radiant_temp, wind_at_height
from otc_ml.pipeline import run_single_index

# Physics (paper Eqs. 1–2):
tmrt = mean_radiant_temp(ta=30.0, tg=35.0, v=1.0)          # globe thermometer
v10  = wind_at_height(v=2.0, measurement_height_m=1.1, target_height_m=10.0)

# One index end to end (BO + training + evaluation):
dataset = build_dataset(seed=42)                # synthetic benchmark
outcome = run_single_index(dataset, "UTCI", n_trials=20)
print(outcome["best_algorithm"], outcome["results"])
```

---

## ⚠️ About the data in this repository

The paper's field dataset (1,168 measurement points, Tabriz summer 2022) is **available from the
corresponding author on request** and is **not** redistributed here. Instead, this repository ships
a **synthetic benchmark** (`otc_ml.data.build_dataset`):

- morphological features are sampled inside the paper's Table 1 ranges
  (orientation 0–180° in six 45° bins, street width 2–38 m, H/W 0.1–3, floors 1–8);
- microclimate variables follow plausible summer daytime relationships
  (Tmrt from Eq. 1, wind adjusted by Eq. 2);
- the UTCI/PET/PMV distributions are **calibrated by quantile mapping so class fractions match
  Table 2** (verified automatically by `verify_table2_fractions`, tolerance ±3%).

Consequently:

- the shipped dataset reproduces the paper's *statistical structure* (class balance, feature ranges,
  learnable morphology→index signal) — it does **not** reproduce its *published metric values*;
- index values are physically motivated approximations, not RayMan/BioKlima reproductions;
- to reproduce the paper's actual Table 3 numbers, obtain the field data from the corresponding
  author and run `python scripts/run_study.py --csv <your.csv>`.

All of this is stated explicitly rather than hidden: no fabricated measurements, no cloned
"results" numbers.

---

## Repository layout

```
tabriz-otc-ml/
├── otc_ml/                  # the package
│   ├── config.py            # paper constants: features, Table 2 classes, search spaces, Table 3
│   ├── features.py          # orientation binning, one-hot encoding, validation
│   ├── data.py              # Eqs. 1–2, synthetic benchmark, CSV loading
│   ├── models.py            # the six classifiers + balanced weighting
│   ├── bayes_opt.py         # GP-EI Bayesian optimization (Optuna)
│   ├── evaluate.py          # weighted-average metrics (Table 3)
│   ├── explain.py           # SHAP per-class importance + directions (Figs. 3–4)
│   └── pipeline.py          # run_single_index / run_study orchestrators
├── scripts/
│   └── run_study.py         # command-line entry point
├── tests/
│   └── test_otc_ml.py       # 28 unit tests incl. physics regression checks
├── data/                    # put your field CSV here
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

## Methodology in one paragraph

For every measurement point, six morphological descriptors (orientation class, street width, H/W
ratio, building floors, tree count) are paired with summer daytime microclimate observations. Mean
radiant temperature is computed with the globe-thermometer equation and wind speed is transported
to pedestrian (1.1 m) and reference (10 m) height with a logarithmic profile. Each comfort index is
binned into the paper's three thermal-stress classes, then six classifiers — KNN, Decision Tree,
SVM, Random Forest, XGBoost and CatBoost — are tuned by Gaussian-Process Bayesian optimization with
Expected Improvement acquisition against a five-fold cross-validated weighted accuracy. The tuned
models are retrained with class-balanced sample weights on 70 % of the data and scored on the
remaining 30 % with weighted average accuracy, precision, recall and F1. Finally, SHAP values of the
best model per index quantify which morphological features drive each thermal-stress class and in
which direction. In the paper this identified 90° orientation, street width and 180° orientation as
the dominant regulators for UTCI and PET, while street width dominated PMV.

## Reproducibility

```bash
python -m unittest discover -s tests -v     # 28 tests
```

Tests cover the physics equations against hand-computed values, Table 2 classification boundaries,
feature validation rules, all six estimator factories, class-balanced weighting, the weighted
metrics, a Bayesian-optimization smoke test, and an end-to-end single-index run. Every entry point
takes a `seed` argument; the default (42) is used throughout.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{alinasab2025measurement,
  author  = {Alinasab, Niloufar and Mohammadzadeh, Negar and Karimi, Alireza and
             Mohammadzadeh, Rahmat and G{\'a}l, Tam{\'a}s},
  title   = {A measurement-based framework integrating machine learning and
             morphological dynamics for outdoor thermal regulation},
  journal = {International Journal of Biometeorology},
  year    = {2025},
  volume  = {69},
  number  = {9},
  pages   = {1645--1662},
  doi     = {10.1007/s00484-025-02921-8}
}
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE). The underlying paper is open access under CC-BY 4.0.

## Authors & contributions

- **Alireza Karimi** ([@Alirezza18](https://github.com/Alirezza18)) — *code author*; implemented
  and maintains this reference implementation (IBBTE, University of Stuttgart).
- **Niloufar Alinasab** ([@nfaralinasab-hub](https://github.com/nfaralinasab-hub)) — *field-study
  lead and corresponding author* of the underlying paper (University of Szeged).

See [CONTRIBUTING.md](CONTRIBUTING.md) to get involved.

## Corresponding author contact

Field-data requests (as stated in the paper's data-availability section) go to the corresponding
author, **Niloufar Alinasab** — `nfar.alinasab@gmail.com`.
