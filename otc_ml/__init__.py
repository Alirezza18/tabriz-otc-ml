"""OTC-ML: ML + Bayesian optimization + SHAP for outdoor thermal comfort.

Reimplementation of the published study:
"A measurement-based framework integrating machine learning and morphological
dynamics for outdoor thermal regulation" (Int J Biometeorol, 2025),
https://doi.org/10.1007/s00484-025-02921-8
"""

from .config import (
    BEST_MODELS,
    CLASS_NAMES,
    FEATURES,
    OTC_CLASS_DEFS,
    SEARCH_SPACES,
    SEED,
)
from .data import (
    INDEX_COLUMNS,
    build_dataset,
    load_dataset,
    mean_radiant_temp,
    verify_table2_fractions,
    wind_at_height,
)
from .evaluate import evaluate_classifier, evaluate_all_models, format_results_table
from .features import (
    build_feature_frame,
    classify_otc,
    class_label_to_int,
    one_hot_orientation,
    orientation_to_bin,
)
from .models import ALGORITHMS, balanced_sample_weights, fit_estimator, make_estimator
from .pipeline import run_single_index, run_study

__version__ = "1.0.0"

__all__ = [
    "ALGORITHMS",
    "BEST_MODELS",
    "CLASS_NAMES",
    "FEATURES",
    "INDEX_COLUMNS",
    "OTC_CLASS_DEFS",
    "SEARCH_SPACES",
    "SEED",
    "__version__",
    "balanced_sample_weights",
    "build_dataset",
    "build_feature_frame",
    "class_label_to_int",
    "classify_otc",
    "evaluate_all_models",
    "evaluate_classifier",
    "fit_estimator",
    "format_results_table",
    "load_dataset",
    "make_estimator",
    "mean_radiant_temp",
    "one_hot_orientation",
    "orientation_to_bin",
    "run_single_index",
    "run_study",
    "verify_table2_fractions",
    "wind_at_height",
]
