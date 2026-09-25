"""End-to-end study pipeline (the paper's full workflow in one call).

Steps, mirroring the paper's research diagram (Fig. 1):
  1. dataset (synthetic benchmark or your own field CSV) -> feature matrix
  2. per OTC index: 70/30 split, class-balanced sample weights
  3. per algorithm: GP-EI Bayesian optimization (5-fold CV objective)
  4. final training of all 6 optimized classifiers
  5. weighted-average metrics on the hold-out split (Table 3)
  6. SHAP analysis of the best model per index (Figs. 3-4 equivalent)
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .bayes_opt import optimize_hyperparameters
from .config import (
    BEST_MODELS,
    CLASS_NAMES,
    FEATURES,
    SEED,
    TEST_SIZE,
)
from .data import INDEX_COLUMNS, build_dataset, load_dataset
from .evaluate import evaluate_all_models, evaluate_classifier, format_results_table
from .explain import explain_model
from .features import build_feature_frame, classify_otc, class_label_to_int
from .models import fit_estimator

__all__ = ["run_study", "run_single_index"]


def _prepare(index_frame: pd.DataFrame, index_name: str):
    """Feature matrix + integer labels for one OTC index."""
    X = index_frame[FEATURES].to_numpy(dtype=float)
    labels = classify_otc(index_name, index_frame[index_name].to_numpy())
    y = np.asarray(class_label_to_int(index_name, labels))
    return X, y


def run_single_index(
    dataset: pd.DataFrame,
    index_name: str,
    n_trials: int = 20,
    algorithms=None,
    seed: int = SEED,
    verbose: bool = False,
) -> Dict:
    """Full BO + training + evaluation for ONE OTC index (UTCI, PET or PMV)."""
    if index_name not in INDEX_COLUMNS:
        raise ValueError(f"index_name must be one of {INDEX_COLUMNS}")

    X, y = _prepare(dataset, index_name)
    class_names = CLASS_NAMES[index_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    algorithms = algorithms or ["KNN", "DT", "SVM", "RF", "XGBoost", "CatBoost"]
    trained: Dict[str, object] = {}
    best_params: Dict[str, dict] = {}

    for algorithm in algorithms:
        t0 = time.perf_counter()
        params = optimize_hyperparameters(algorithm, X_train, y_train, n_trials=n_trials, seed=seed)
        estimator = fit_estimator(algorithm, params, X_train, y_train)
        trained[algorithm] = estimator
        best_params[algorithm] = params
        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"  [{index_name}] {algorithm}: BO + fit in {elapsed:.1f}s")

    rows = evaluate_all_models(trained, X_test, y_test)
    best_algorithm = rows[0]["algorithm"]

    return {
        "index": index_name,
        "class_names": class_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "trained": trained,
        "best_params": best_params,
        "results": rows,
        "best_algorithm": best_algorithm,
        "best_estimator": trained[best_algorithm],
    }


def run_study(
    dataset: Optional[pd.DataFrame] = None,
    csv_path: Optional[str] = None,
    n_trials: int = 20,
    with_shap: bool = True,
    seed: int = SEED,
    verbose: bool = False,
) -> Dict:
    """Run the complete three-index study.

    Provide either a prepared DataFrame (with morphological features and the
    three OTC index columns) or a `csv_path` (see `data.load_dataset`).
    When neither is given, the synthetic Tabriz-summer benchmark is used.
    """
    if dataset is None and csv_path is None:
        dataset = build_dataset(seed=seed)
    elif dataset is None:
        dataset = load_dataset(csv_path)

    study = {"indices": {}, "tables": {}, "shap": {}}

    for index_name in INDEX_COLUMNS:
        if verbose:
            print(f"=== {index_name} ===")
        outcome = run_single_index(dataset, index_name, n_trials=n_trials, seed=seed, verbose=verbose)
        study["indices"][index_name] = outcome
        study["tables"][index_name] = format_results_table(outcome["results"], index_name)

        if with_shap:
            shap_out = explain_model(outcome["best_estimator"], outcome["X_test"], index_name)
            study["shap"][index_name] = shap_out

    study["best_models"] = {
        index: outcome["best_algorithm"] for index, outcome in study["indices"].items()
    }
    study["expected_best_models"] = dict(BEST_MODELS)
    return study
