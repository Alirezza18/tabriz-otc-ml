"""Bayesian optimization of the six classifiers (paper "ML optimization process").

Implements exactly what the paper describes:
- a Gaussian-Process surrogate over the hyperparameter space (Optuna's
  `GP` sampler with `IndependentNUTS`-style GP fitting behind the scenes);
- Expected Improvement (EI) as the acquisition function;
- fivefold cross-validated weighted accuracy as the objective;
- log-uniform sampling for learning-rate-type parameters, uniform/integer
  sampling for structural ones;
- a fixed seed so the whole study is reproducible.
"""

from __future__ import annotations

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from .config import BO_N_TRIALS, BO_SEED, CV_FOLDS, SEARCH_SPACES
from .models import SUPPORTS_SAMPLE_WEIGHT, balanced_sample_weights, predict_labels

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_params(trial: optuna.Trial, algorithm: str) -> dict:
    """Sample one hyperparameter configuration for `algorithm` from its space."""
    space = SEARCH_SPACES[algorithm]
    params: dict = {}

    for name, spec in space.items():
        if spec[0] == "log":
            # ["log", low, high] -> log-uniform continuous parameter
            _, low, high = spec
            params[name] = trial.suggest_float(name, low, high, log=True)
        elif isinstance(spec[0], str):
            params[name] = trial.suggest_categorical(name, spec)
        elif any(v is None for v in spec):
            # Structural integer parameters whose value may be None (e.g.
            # max_depth=None = unlimited). Optuna cannot mix None into an
            # int range, so sample the int range and map one value to None.
            finite = [v for v in spec if v is not None]
            low, high = min(finite), max(finite)
            value = trial.suggest_int(name + "_finite", low, high)
            use_none = trial.suggest_categorical(name + "_unlimited", [True, False])
            params[name] = None if use_none else value
        elif all(isinstance(v, int) for v in spec):
            params[name] = trial.suggest_int(name, spec[0], spec[-1])
        else:
            params[name] = trial.suggest_float(name, spec[0], spec[-1])

    return params


def optimize_hyperparameters(
    algorithm: str,
    X_train,
    y_train,
    n_trials: int = BO_N_TRIALS,
    seed: int = BO_SEED,
    n_jobs_cv: int = 1,
) -> dict:
    """Run GP-EI Bayesian optimization for one algorithm.

    Returns the best hyperparameter dictionary (ready for `fit_estimator`).
    The BO objective is the fivefold cross-validated weighted accuracy,
    exactly the metric the paper reports and the paper says BO maximizes.
    """
    X_arr = np.asarray(X_train)
    y_arr = np.asarray(y_train)
    weights = balanced_sample_weights(y_arr)
    use_weights = algorithm in SUPPORTS_SAMPLE_WEIGHT

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, algorithm)
        estimator = _build(algorithm, params)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        scores = []
        for train_idx, valid_idx in cv.split(X_arr, y_arr):
            fold_model = clone(estimator)
            if use_weights:
                fold_model.fit(
                    X_arr[train_idx], y_arr[train_idx], sample_weight=weights[train_idx]
                )
            else:
                fold_model.fit(X_arr[train_idx], y_arr[train_idx])
            predictions = predict_labels(fold_model, X_arr[valid_idx], y_arr)
            # Weighted (balanced) accuracy of the fold - the paper's BO target.
            fold_weights = weights[valid_idx]
            hits = (predictions == y_arr[valid_idx]).astype(float)
            scores.append(float(np.average(hits, weights=fold_weights)))
        return float(np.mean(scores))

    sampler = optuna.samplers.GPSampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    # Fold structural-int placeholders (name_finite / name_unlimited, used
    # because Optuna cannot mix None into an int range) back into their
    # original parameter names, e.g. max_depth=None.
    for key in [k for k in list(best) if k.endswith("_finite")]:
        base = key[: -len("_finite")]
        value = best.pop(key)
        use_none = best.pop(base + "_unlimited", False)
        best[base] = None if use_none else value
    # Re-attach non-suggested structural values that the estimator needs.
    if algorithm == "SVM" and "kernel" not in best:
        best["kernel"] = "rbf"
    if algorithm in ("DT", "RF") and "criterion" not in best:
        best["criterion"] = "gini"
    return best


def _build(algorithm: str, params: dict):
    """Build a bare estimator for BO trials (no balanced weights plumbing)."""
    from .models import make_estimator

    if algorithm == "SVM":
        params = {**params, "probability": False}  # faster inside BO folds
        return make_estimator(algorithm, params)
    return make_estimator(algorithm, params)


def optimize_all(
    X_train,
    y_train,
    algorithms=None,
    n_trials: int = BO_N_TRIALS,
    seed: int = BO_SEED,
    verbose: bool = False,
) -> dict:
    """Optimize every algorithm; returns {algorithm: best_params}."""
    algorithms = algorithms or list(SEARCH_SPACES.keys())
    results: dict = {}
    for algorithm in algorithms:
        if verbose:
            print(f"[BO] optimizing {algorithm} ({n_trials} trials)...")
        results[algorithm] = optimize_hyperparameters(
            algorithm, X_train, y_train, n_trials=n_trials, seed=seed
        )
    return results
