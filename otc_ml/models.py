"""The six machine-learning classifiers studied in the paper.

KNN, Decision Tree, SVM, Random Forest, XGBoost and CatBoost, exactly as in
the paper's "ML optimization process" section. Each factory returns a fresh,
deterministically-seeded estimator; hyperparameters are supplied by the
Bayesian optimizer (see `bayes_opt.py`) or fall back to sensible defaults.

Class imbalance is handled the way the paper describes it: sample weights
computed with sklearn's `compute_sample_weight(class_weight="balanced")`,
which up-weights under-represented thermal-stress classes.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from .config import SEED

ALGORITHMS = ["KNN", "DT", "SVM", "RF", "XGBoost", "CatBoost"]

# KNeighborsClassifier.fit() takes no sample_weight; the other five do.
SUPPORTS_SAMPLE_WEIGHT = {"DT", "SVM", "RF", "XGBoost", "CatBoost"}


def make_estimator(algorithm: str, params: Dict | None = None):
    """Instantiate one of the six algorithms with the given hyperparameters."""
    params = dict(params or {})

    if algorithm == "KNN":
        defaults = {"n_neighbors": 7, "weights": "distance", "p": 2}
        defaults.update(params)
        return KNeighborsClassifier(**defaults)

    if algorithm == "DT":
        defaults = {
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "criterion": "gini",
            "random_state": SEED,
        }
        defaults.update(params)
        return DecisionTreeClassifier(**defaults)

    if algorithm == "SVM":
        defaults = {"C": 10.0, "gamma": "scale", "kernel": "rbf", "random_state": SEED}
        defaults.update(params)
        # Caller may override (BO passes probability=False for speed).
        defaults.setdefault("probability", True)
        return SVC(**defaults)

    if algorithm == "RF":
        defaults = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": SEED,
        }
        defaults.update(params)
        return RandomForestClassifier(**defaults)

    if algorithm == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "XGBoost is required for the XGBoost model. "
                "Install it with: pip install xgboost"
            ) from exc
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "random_state": SEED,
            "n_jobs": -1,
        }
        defaults.update(params)
        return XGBClassifier(**defaults)

    if algorithm == "CatBoost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "CatBoost is required for the CatBoost model. "
                "Install it with: pip install catboost"
            ) from exc
        defaults = {
            "iterations": 500,
            "learning_rate": 0.1,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "loss_function": "MultiClass",
            "verbose": False,
            "allow_writing_files": False,
            "random_seed": SEED,
            "thread_count": -1,
        }
        defaults.update(params)
        return CatBoostClassifier(**defaults)

    raise ValueError(f"unknown algorithm '{algorithm}'; expected one of {ALGORITHMS}")


def balanced_sample_weights(y_train: np.ndarray) -> np.ndarray:
    """Paper's imbalance handling: compute_sample_weight('balanced')."""
    return compute_sample_weight(class_weight="balanced", y=y_train)


def fit_estimator(algorithm: str, params: Dict | None, X_train, y_train):
    """Fit one estimator with the paper's class-balanced sample weights.

    The paper applies `compute_sample_weight(class_weight='balanced')` to
    counter class imbalance; KNN has no weighted fit in scikit-learn, so it
    is trained unweighted (its performance gap is part of the paper's
    findings anyway).
    """
    estimator = make_estimator(algorithm, params)
    if algorithm in SUPPORTS_SAMPLE_WEIGHT:
        weights = balanced_sample_weights(np.asarray(y_train))
        estimator.fit(X_train, y_train, sample_weight=weights)
    else:
        estimator.fit(X_train, y_train)
    return estimator


def predict_labels(estimator, X, y_ref=None) -> np.ndarray:
    """Predict class labels, normalized to a flat array of the reference dtype.

    Library quirk this guards against: CatBoost's ``predict`` returns a
    column vector of shape ``(n, 1)`` (and sometimes float labels). Compared
    naively against a 1-D truth array it broadcasts into an ``(n, n)``
    hit-matrix, which silently corrupts every accuracy/F1 computation
    (e.g. 20008/40000 = 0.5002 for n=200). Always route predictions through
    this helper instead of calling ``estimator.predict`` directly.
    """
    predictions = np.asarray(estimator.predict(X)).reshape(-1)
    if y_ref is not None:
        y_ref = np.asarray(y_ref).reshape(-1)
        if np.issubdtype(y_ref.dtype, np.integer):
            predictions = predictions.astype(y_ref.dtype)
    return predictions
