"""Central configuration for the OTC-ML framework.

Constants here trace back to the published study:
"A measurement-based framework integrating machine learning and morphological
dynamics for outdoor thermal regulation" (Int J Biometeorol, 2025),
https://doi.org/10.1007/s00484-025-02921-8
"""

from __future__ import annotations

from typing import Dict, List, Tuple

SEED = 42

# ---------------------------------------------------------------------------
# Morphological feature layout (paper Sections "Data collecting process" and
# "Data pre-processing and classification", Table 1)
# ---------------------------------------------------------------------------
FEATURES: List[str] = [
    "orientation_45",
    "orientation_90",
    "orientation_135",
    "orientation_180",
    "street_width_m",
    "hw_ratio",
    "n_floors",
    "n_trees",
]

# One-hot orientation classes (45-degree bins per Table 1). "orientation_0"
# (0 degrees / true North) is the reference level and is not encoded as a
# column, to avoid the dummy-variable trap.
ORIENTATION_BINS: List[Tuple[str, float, float]] = [
    ("orientation_45", 0.0, 45.0),    # 0-45 degrees
    ("orientation_90", 45.0, 90.0),   # 45-90 degrees
    ("orientation_135", 90.0, 135.0), # 90 degrees / 90-135 degrees
    ("orientation_180", 135.0, 180.0) # 135-180 degrees / 180 degrees
]

HW_CLASSES: List[str] = ["0.1-1", "1.1-2", "2.1-3"]
WIDTH_RANGE_M: Tuple[float, float] = (2.0, 38.0)   # "street width ranges from 2 to 38 m"
FLOORS_RANGE: Tuple[int, int] = (1, 8)             # "building elevation differs from 1 to 8 floors"
HW_RATIO_RANGE: Tuple[float, float] = (0.1, 3.0)

# ---------------------------------------------------------------------------
# OTC index classification into 3 thermal-stress classes (paper Table 2)
# Each entry: (lower bound, upper bound, class name, dataset fraction)
# ---------------------------------------------------------------------------
OTC_CLASS_DEFS: Dict[str, List[Tuple[float, float, str, float]]] = {
    "UTCI": [
        (26.0, 32.0, "Moderate heat stress", 0.28),
        (32.0, 38.0, "Strong heat stress", 0.32),
        (38.0, 46.0, "Very strong heat stress", 0.40),
    ],
    "PET": [
        (29.0, 35.0, "Moderate heat stress", 0.21),
        (35.0, 41.0, "Strong heat stress", 0.20),
        (41.0, 60.0, "Extreme heat stress", 0.59),
    ],
    "PMV": [
        (2.0, 3.0, "Warm", 0.28),
        (3.0, 4.0, "Hot", 0.22),
        (4.0, 7.0, "Very hot", 0.50),
    ],
}

# Class fractions reported in Table 2 (used to size the synthetic benchmark so
# its class balance mirrors the published dataset).
CLASS_FRACTIONS: Dict[str, List[float]] = {
    index: [entry[3] for entry in entries]
    for index, entries in OTC_CLASS_DEFS.items()
}

CLASS_NAMES: Dict[str, List[str]] = {
    index: [entry[2] for entry in entries]
    for index, entries in OTC_CLASS_DEFS.items()
}

# ---------------------------------------------------------------------------
# Physical equations (paper Eqs. 1-2)
# ---------------------------------------------------------------------------
GLOBE_DIAMETER_M = 0.150   # D in Eq. 1
GLOBE_EMISSIVITY = 0.95    # epsilon in Eq. 1
URBAN_Z0_M = 0.01          # z0 in Eq. 2 (urban terrain)

# Standard person settings for the comfort indices (paper, "Measurement
# procedure"): PET metabolic rate 80 W, UTCI metabolic rate 135 W/m2.
PET_METABOLIC_W = 80.0
UTCI_METABOLIC_WM2 = 135.0

# ---------------------------------------------------------------------------
# Machine-learning setup (paper Sections "ML optimization process" and
# "Model performance")
# ---------------------------------------------------------------------------
TEST_SIZE = 0.30           # 70/30 train/test split
CV_FOLDS = 5               # fivefold cross-validation inside BO
BO_N_TRIALS = 60           # BO iterations per model/index pair
BO_ACQ_FUNC = "EI"         # Expected Improvement acquisition function
BO_SEED = SEED

# Hyperparameter search spaces (log-uniform for learning-type parameters,
# uniform/integer for structural ones, per the paper's description).
SEARCH_SPACES: Dict[str, Dict[str, List]] = {
    "KNN": {
        "n_neighbors": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 20],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "DT": {
        "max_depth": [3, 4, 5, 6, 7, 8, 10, 12, None],
        "min_samples_split": [2, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 5, 8],
        "criterion": ["gini", "entropy"],
    },
    "SVM": {
        "C": ["log", 1e-1, 1e3],
        "gamma": ["log", 1e-4, 1e0],
        "kernel": ["rbf"],
    },
    "RF": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 5, 8, 12, 16, 20],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 3, 5],
        "max_features": ["sqrt", "log2", 0.5],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300, 400, 600],
        "learning_rate": ["log", 1e-2, 3e-1],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 5, 10],
    },
    "CatBoost": {
        "iterations": [200, 300, 500, 800],
        "learning_rate": ["log", 1e-2, 3e-1],
        "depth": [4, 5, 6, 7, 8],
        "l2_leaf_reg": ["log", 1.0, 10.0],
    },
}

# Paper Table 3, headline results (weighted metrics on the 30% holdout).
EXPECTED_RESULTS: Dict[str, Dict[str, Dict[str, float]]] = {
    "UTCI": {
        "KNN": {"accuracy": 71, "precision": 72, "recall": 71, "f1": 71},
        "DT": {"accuracy": 71, "precision": 71, "recall": 71, "f1": 71},
        "SVM": {"accuracy": 67, "precision": 71, "recall": 67, "f1": 66},
        "RF": {"accuracy": 69, "precision": 69, "recall": 69, "f1": 68},
        "XGBoost": {"accuracy": 70, "precision": 70, "recall": 70, "f1": 70},
        "CatBoost": {"accuracy": 72, "precision": 73, "recall": 72, "f1": 72},
    },
    "PET": {
        "KNN": {"accuracy": 71, "precision": 69, "recall": 71, "f1": 69},
        "DT": {"accuracy": 74, "precision": 74, "recall": 74, "f1": 73},
        "SVM": {"accuracy": 72, "precision": 70, "recall": 72, "f1": 70},
        "RF": {"accuracy": 76, "precision": 75, "recall": 76, "f1": 75},
        "XGBoost": {"accuracy": 69, "precision": 69, "recall": 69, "f1": 69},
        "CatBoost": {"accuracy": 75, "precision": 74, "recall": 75, "f1": 74},
    },
    "PMV": {
        "KNN": {"accuracy": 69, "precision": 69, "recall": 69, "f1": 69},
        "DT": {"accuracy": 71, "precision": 71, "recall": 71, "f1": 70},
        "SVM": {"accuracy": 71, "precision": 71, "recall": 71, "f1": 69},
        "RF": {"accuracy": 75, "precision": 74, "recall": 75, "f1": 74},
        "XGBoost": {"accuracy": 83, "precision": 83, "recall": 83, "f1": 82},
        "CatBoost": {"accuracy": 80, "precision": 81, "recall": 80, "f1": 79},
    },
}

# Best model per OTC index, as reported in the paper.
BEST_MODELS: Dict[str, str] = {
    "UTCI": "CatBoost",
    "PET": "RF",
    "PMV": "XGBoost",
}
