"""Feature engineering and OTC index classification.

Implements the paper's preprocessing decisions:
- one-hot encoding of orientation into 45-degree bins (Table 1);
- 3-class binning of each OTC index per Table 2;
- morphological feature vectors combining orientation, width, H/W ratio,
  building floors and tree count.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    CLASS_NAMES,
    FEATURES,
    HW_RATIO_RANGE,
    OTC_CLASS_DEFS,
    ORIENTATION_BINS,
    WIDTH_RANGE_M,
)

# ---------------------------------------------------------------------------
# Orientation binning
# ---------------------------------------------------------------------------

def orientation_to_bin(angle_deg: float) -> str:
    """Map a street orientation angle (0-180 degrees) to its one-hot column.

    Bins follow Table 1's 45-degree classification. Angles are normalized to
    [0, 180) first; exact boundary values fall in the lower bin they close.
    """
    angle = float(angle_deg) % 180.0
    if angle == 0.0 and float(angle_deg) % 360.0 == 180.0:
        # Exact 180 degrees (E-W) normalizes to 0 via the modulo; restore it
        # so it lands in the paper's explicit 180-degree class.
        angle = 180.0
    for name, lo, hi in ORIENTATION_BINS:
        if lo <= angle < hi:
            return name
    return ORIENTATION_BINS[-1][0]


def one_hot_orientation(angles_deg: Sequence[float]) -> pd.DataFrame:
    """One-hot encode street orientations into the 4 Table-1 bins."""
    records = []
    for angle in angles_deg:
        bin_name = orientation_to_bin(angle)
        records.append({feature: int(feature == bin_name) for feature in FEATURES[:4]})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# OTC index classification (Table 2)
# ---------------------------------------------------------------------------

def classify_otc(index_name: str, values: Sequence[float]) -> List[str]:
    """Assign Table-2 class names to raw index values.

    Values below the lowest published class bound are clipped into the lowest
    class, mirroring the paper's note that summer data below the comfort range
    was removed/rebinned, leaving a 3-class problem.
    """
    classes = OTC_CLASS_DEFS[index_name]
    labels: List[str] = []
    for value in values:
        label = None
        for lo, hi, name, _ in classes:
            if lo <= value < hi:
                label = name
                break
        if label is None:
            # Below the first lower bound or above the last upper bound:
            # clip to the nearest class (the paper drops sub-comfort values,
            # but the benchmark generator never produces them; clipping keeps
            # this function total for user-supplied data too).
            label = classes[0][2] if value < classes[0][0] else classes[-1][2]
        labels.append(label)
    return labels


def class_label_to_int(index_name: str, labels: Sequence[str]) -> List[int]:
    """Map class names to 0/1/2 in Table-2 order (low to high stress)."""
    names = CLASS_NAMES[index_name]
    lookup = {name: idx for idx, name in enumerate(names)}
    return [lookup[label] for label in labels]


# ---------------------------------------------------------------------------
# Assembling feature frames
# ---------------------------------------------------------------------------

def build_feature_frame(
    orientation_deg: Sequence[float],
    street_width_m: Sequence[float],
    hw_ratio: Sequence[float],
    n_floors: Sequence[float],
    n_trees: Sequence[float],
) -> pd.DataFrame:
    """Assemble the morphological feature matrix used by every model.

    Layout: 4 one-hot orientation columns + street width + H/W ratio +
    number of floors + number of trees (paper's five recorded morphological
    characteristics; orientation expanded per Table 1).
    """
    orientation_df = one_hot_orientation(orientation_deg)
    continuous = pd.DataFrame(
        {
            "street_width_m": np.asarray(street_width_m, dtype=float),
            "hw_ratio": np.asarray(hw_ratio, dtype=float),
            "n_floors": np.asarray(n_floors, dtype=float),
            "n_trees": np.asarray(n_trees, dtype=float),
        }
    )
    return pd.concat([orientation_df.reset_index(drop=True), continuous], axis=1)


def validate_feature_frame(frame: pd.DataFrame) -> None:
    """Raise if a user-supplied feature frame deviates from the paper layout."""
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"feature frame is missing required columns: {missing}")
    extra = [column for column in frame.columns if column not in FEATURES]
    if extra:
        raise ValueError(f"feature frame has unexpected columns: {extra}")
    if frame.empty:
        raise ValueError("feature frame is empty")

    if not frame[FEATURES[:4]].isin([0, 1]).all().all():
        raise ValueError("orientation one-hot columns must be 0/1")
    sums = frame[FEATURES[:4]].sum(axis=1)
    if not (sums == 1).all():
        raise ValueError("exactly one orientation bin must be active per row")
    for column in ["street_width_m", "hw_ratio", "n_floors", "n_trees"]:
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"column '{column}' contains non-finite values")
    if frame["street_width_m"].min() < WIDTH_RANGE_M[0] - 1e-9:
        raise ValueError("street width below the published 2 m minimum")
    if frame["street_width_m"].max() > WIDTH_RANGE_M[1] + 1e-9:
        raise ValueError("street width above the published 38 m maximum")
    if frame["hw_ratio"].min() < HW_RATIO_RANGE[0] - 1e-9:
        raise ValueError("H/W ratio below the published 0.1 minimum")


# Backwards-compatible aliases used by the pipeline tests.
def classify_index(index_name: str, values: Sequence[float]) -> List[str]:
    return classify_otc(index_name, values)


def build_features(
    orientation_deg: Sequence[float],
    street_width_m: Sequence[float],
    hw_ratio: Sequence[float],
    n_floors: Sequence[float],
    n_trees: Sequence[float],
) -> pd.DataFrame:
    return build_feature_frame(
        orientation_deg, street_width_m, hw_ratio, n_floors, n_trees
    )


def make_demo_frame(n_points: int = 1168, seed: int = 42) -> pd.DataFrame:
    """Small helper for demos: random morphologies inside the published ranges."""
    rng = np.random.default_rng(seed)
    orientations = rng.choice([0.0, 30.0, 60.0, 100.0, 150.0, 180.0], size=n_points)
    widths = rng.uniform(*WIDTH_RANGE_M, size=n_points)
    hw = rng.uniform(*HW_RATIO_RANGE, size=n_points)
    floors = rng.integers(1, 9, size=n_points).astype(float)
    trees = rng.integers(0, 40, size=n_points).astype(float)
    return build_feature_frame(orientations, widths, hw, floors, trees)
