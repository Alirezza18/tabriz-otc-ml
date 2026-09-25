"""SHAP interpretability (paper Section "Model interpretation").

Two analyses, mirroring the paper's Figures 3 and 4:

1. Per-class feature importance: mean |SHAP| aggregated per (feature, class)
   pair - the paper's Fig. 3 ranking of the morphological features per
   thermal-stress class.
2. Binary SHAP direction analysis: for each feature and class, whether high
   feature values push the prediction toward (+) or away from (-) that class -
   the paper's Fig. 4 "red = larger values, blue = smaller values" reading.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import CLASS_NAMES, FEATURES

TREE_MODELS = ("RandomForestClassifier", "XGBClassifier", "CatBoostClassifier")


def compute_shap_values(estimator, X) -> Tuple[np.ndarray, np.ndarray]:
    """Return (shap_values, X_explained) for a fitted estimator.

    Tree-based models (RF, XGBoost, CatBoost) use the fast TreeExplainer;
    KNN / DT / SVM fall back to the model-agnostic PermutationExplainer so
    every one of the six algorithms can be explained. shap_values has shape
    (n_samples, n_features, n_classes).
    """
    import shap

    X_arr = np.asarray(X, dtype=float)
    model_name = type(estimator).__name__

    if model_name in TREE_MODELS:
        explanation = shap.TreeExplainer(estimator)(X_arr)
    else:
        background = shap.maskers.Independent(X_arr, max_samples=min(100, len(X_arr)))
        explainer = shap.Explainer(
            estimator.predict_proba, background, algorithm="permutation"
        )
        explanation = explainer(X_arr[: min(200, len(X_arr))])

    values = np.asarray(explanation.values)
    if values.ndim == 2:
        # Some explainers squeeze the class axis for binary problems; the
        # study is 3-class, but keep a defensive unsqueeze.
        values = values[:, :, None]
    return values, np.asarray(explanation.data, dtype=float)


def per_class_importance(
    shap_values: np.ndarray,
    class_names: List[str],
) -> pd.DataFrame:
    """Mean |SHAP| per (feature, class) -> long-format DataFrame.

    Output columns: class, feature, mean_abs_shap, rank (1 = most influential
    within the class). Equivalent to the paper's Fig. 3 rankings.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)  # (n_features, n_classes)
    records = []
    for class_idx, class_name in enumerate(class_names):
        feats = [
            (FEATURES[i], float(mean_abs[i, class_idx]))
            for i in range(mean_abs.shape[0])
        ]
        feats.sort(key=lambda item: item[1], reverse=True)
        for rank, (feature, value) in enumerate(feats, start=1):
            records.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "mean_abs_shap": round(value, 5),
                    "rank": rank,
                }
            )
    return pd.DataFrame(records)


def binary_shap_directions(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
) -> pd.DataFrame:
    """Sign correlation between feature values and SHAP contributions.

    For every (feature, class): the Pearson correlation between the feature's
    value and its SHAP value across samples. Positive correlation means
    "larger feature values push toward this class" (red in the paper's Fig. 4);
    negative means "larger values push away" (blue).
    """
    records = []
    for f_idx, feature in enumerate(feature_names):
        x_vals = X[:, f_idx]
        x_std = x_vals.std()
        for c_idx, class_name in enumerate(class_names):
            shap_vals = shap_values[:, f_idx, c_idx]
            corr = (
                float(np.corrcoef(x_vals, shap_vals)[0, 1]) if x_std > 1e-12 else 0.0
            )
            records.append(
                {
                    "feature": feature,
                    "class": class_name,
                    "value_shap_correlation": round(corr, 4),
                    "direction": (
                        "toward class"
                        if corr > 0.1
                        else ("away from class" if corr < -0.1 else "neutral")
                    ),
                }
            )
    return pd.DataFrame(records)


def explain_model(estimator, X, index_name: str) -> Dict[str, pd.DataFrame]:
    """One-call SHAP analysis for a fitted estimator.

    Returns:
        {
          "per_class_importance": long DataFrame (Fig. 3 equivalent),
          "binary_directions":    long DataFrame (Fig. 4 equivalent),
        }
    """
    shap_values, X_explained = compute_shap_values(estimator, X)
    class_names = CLASS_NAMES[index_name]
    importance = per_class_importance(shap_values, class_names)
    directions = binary_shap_directions(
        shap_values, X_explained, FEATURES, class_names
    )
    return {"per_class_importance": importance, "binary_directions": directions}
