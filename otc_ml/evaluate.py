"""Model evaluation with the paper's weighted-average protocol.

The paper evaluates every classifier with the *weighted average* of accuracy,
precision, recall and F1 across the three thermal-stress classes, so that
class imbalance does not skew the comparison. All metrics are computed on the
30 % hold-out test split ("Model performance" section, Table 3).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .models import predict_labels

METRICS = ["accuracy", "precision", "recall", "f1"]


def evaluate_classifier(y_true, y_pred) -> Dict[str, float]:
    """Weighted-average accuracy / precision / recall / F1, in percent."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "accuracy": round(100.0 * accuracy_score(y_true, y_pred), 1),
        "precision": round(
            100.0 * precision_score(y_true, y_pred, average="weighted", zero_division=0), 1
        ),
        "recall": round(100.0 * recall_score(y_true, y_pred, average="weighted", zero_division=0), 1),
        "f1": round(100.0 * f1_score(y_true, y_pred, average="weighted", zero_division=0), 1),
    }


def evaluate_all_models(
    trained: Dict[str, object],
    X_test,
    y_test,
) -> "list[dict]":
    """Evaluate every trained estimator; returns one Table-3-style row per model.

    Rows: {"algorithm", "accuracy", "precision", "recall", "f1"} with values
    in percent, sorted by f1 descending (best model first).
    """
    rows: List[dict] = []
    for algorithm, estimator in trained.items():
        y_pred = predict_labels(estimator, X_test, y_test)
        metrics = evaluate_classifier(y_test, y_pred)
        rows.append({"algorithm": algorithm, **metrics})
    rows.sort(key=lambda row: row["f1"], reverse=True)
    return rows


def format_results_table(rows: List[dict], index_name: str = "") -> str:
    """Render rows as a paper-Table-3-style markdown table."""
    header = "| Algorithm | Accuracy % | Precision % | Recall % | F1 % |"
    separator = "|---|---|---|---|---|"
    body = "\n".join(
        f"| {row['algorithm']} | {row['accuracy']} | {row['precision']} "
        f"| {row['recall']} | {row['f1']} |"
        for row in rows
    )
    title = f"\n**{index_name}**\n\n" if index_name else "\n"
    return f"{title}{header}\n{separator}\n{body}\n"
