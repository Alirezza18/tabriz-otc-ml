"""Microclimate data utilities.

Two things live here:

1. The paper's physical equations, verbatim:
   - Eq. 1: mean radiant temperature from globe temperature (ISO 7726-style
     globe formula, with the paper's D = 150 mm and eps = 0.95);
   - Eq. 2: log wind profile used to extrapolate wind speed from the
     measurement height to the 1.1 m (PET) and 10 m (UTCI) reference heights,
     with the paper's urban roughness z0 = 0.01 m.

2. A synthetic benchmark generator that reproduces the *statistical shape* of
   the published Tabriz summer dataset (173 streets / 1168 points): features
   drawn inside the Table 1 ranges, microclimate variables drawn from
   summer-in-Tabriz distributions, and OTC index values calibrated so the
   class frequencies match Table 2 (28/32/40 % UTCI, 21/20/59 % PET,
   28/22/50 % PMV).

   IMPORTANT: the synthetic indices are physically-motivated approximations,
   not RayMan / BioKlima 2.6 reproductions. They exist so the pipeline can be
   run, tested and demonstrated end-to-end without the (on-request) field
   dataset. Replace them with your own measurements via `load_dataset` /
   `build_dataset` to reproduce the paper's actual numbers.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    CLASS_FRACTIONS,
    FLOORS_RANGE,
    GLOBE_DIAMETER_M,
    GLOBE_EMISSIVITY,
    HW_RATIO_RANGE,
    SEED,
    URBAN_Z0_M,
    WIDTH_RANGE_M,
)
from .features import build_feature_frame

MICROCLIMATE_COLUMNS = ["Ta", "RH", "V", "Tmrt"]
INDEX_COLUMNS = ["UTCI", "PET", "PMV"]

# Quantile anchors used to calibrate each index so its class shares match
# Table 2: (class-1 share, upper bound of class 1, class-1+2 share, upper
# bound of class 2). E.g. UTCI is 28 % moderate (boundary at 32) and 28+32=60
# % at-or-below strong (boundary at 38).
_CALIBRATION: Dict[str, Tuple[float, float, float, float]] = {
    "UTCI": (0.28, 32.0, 0.60, 38.0),
    "PET": (0.21, 35.0, 0.41, 41.0),
    "PMV": (0.28, 3.0, 0.50, 4.0),
}


# ---------------------------------------------------------------------------
# Physical equations (paper Eqs. 1-2)
# ---------------------------------------------------------------------------

def mean_radiant_temp(
    ta_c: float,
    tglobe_c: float,
    v_m_s: float,
    diameter_m: float = GLOBE_DIAMETER_M,
    emissivity: float = GLOBE_EMISSIVITY,
) -> float:
    """Eq. 1 - mean radiant temperature (C) from globe temperature (C).

    TMRT = [ (Tg + 273.15)^4
             + (1.1e8 * V^0.6) / (eps * D^0.4) * (Tg - Ta) ]^(1/4) - 273.15
    """
    ta = np.asarray(ta_c, dtype=float)
    tg = np.asarray(tglobe_c, dtype=float)
    v = np.clip(np.asarray(v_m_s, dtype=float), 0.01, None)  # avoid V^0.6 = 0 singularity
    term_radiative = (tg + 273.15) ** 4
    term_convective = (1.1e8 * v ** 0.6) / (emissivity * diameter_m ** 0.4) * (tg - ta)
    tmrt = (term_radiative + term_convective) ** 0.25 - 273.15
    return float(tmrt) if np.isscalar(ta_c) and np.isscalar(tglobe_c) else tmrt


def wind_at_height(
    v_measured: float,
    measurement_height_m: float,
    target_height_m: float,
    z0_m: float = URBAN_Z0_M,
) -> float:
    """Eq. 2 - log wind profile extrapolation.

    Vx = Vm * log(x / z0) / log(m / z0), with z0 = 0.01 m for urban terrain.
    """
    vm = np.asarray(v_measured, dtype=float)
    m_height = np.asarray(measurement_height_m, dtype=float)
    x_target = np.asarray(target_height_m, dtype=float)
    vx = vm * np.log(x_target / z0_m) / np.log(m_height / z0_m)
    return float(vx) if np.isscalar(v_measured) else vx


# ---------------------------------------------------------------------------
# Synthetic benchmark generator
# ---------------------------------------------------------------------------

def _microclimate_from_morphology(
    orientation_deg: np.ndarray,
    width: np.ndarray,
    hw: np.ndarray,
    floors: np.ndarray,
    trees: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw summer-Tabriz microclimate variables shaped by morphology."""
    n = len(orientation_deg)

    # Air temperature: July mean maximum ~31.5 C, spikes to 38.2 C (paper).
    ta = rng.normal(loc=31.5, scale=2.2, size=n)
    ta = np.clip(ta, 26.0, 40.0)

    # Relative humidity: semi-arid summer, lower half of the 42-69 % annual band.
    rh = np.clip(rng.normal(loc=30.0, scale=8.0, size=n), 12.0, 60.0)

    # Wind at 1.1 m: light summer air, occasional gusts.
    v = np.clip(rng.lognormal(mean=0.1, sigma=0.55, size=n), 0.2, 5.0)

    # Mean radiant temperature: driven by solar exposure of the canyon.
    #   exposure in [0, 1]: 0 = fully shaded canyon, 1 = fully sunlit.
    orientation_exposure = np.select(
        [
            (orientation_deg >= 0.0) & (orientation_deg < 45.0),    # N-ish
            (orientation_deg >= 45.0) & (orientation_deg < 90.0),   # NE-E
            (orientation_deg >= 90.0) & (orientation_deg < 135.0),  # E-S (incl. 90 N-S)
            (orientation_deg >= 135.0),                             # S-W (incl. 180 E-W)
        ],
        [0.35, 0.55, 0.45, 0.70],
        default=0.5,
    )
    geometry_shade = np.clip(0.30 * hw + 0.012 * trees, 0.0, 0.55)
    width_effect = 0.006 * (width - WIDTH_RANGE_M[0])  # wider street: more sun
    exposure = np.clip(orientation_exposure + width_effect - geometry_shade, 0.05, 1.0)
    tmrt = ta + 12.0 + 38.0 * exposure + rng.normal(0.0, 1.5, size=n)

    return pd.DataFrame({"Ta": ta, "RH": rh, "V": v, "Tmrt": tmrt})


def _approx_indices(microclimate: pd.DataFrame) -> pd.DataFrame:
    """Physically-motivated approximations of UTCI / PET / PMV.

    Different indices weight radiation, wind and humidity differently, as in
    the published literature (UTCI most wind- and radiation-sensitive; PET
    radiation-dominated; PMV intermediate). Values are later quantile-
    calibrated to the Table 2 boundaries, so only the *ordering* structure
    matters here, not absolute degrees.
    """
    ta = microclimate["Ta"].to_numpy()
    rh = microclimate["RH"].to_numpy()
    v = microclimate["V"].to_numpy()
    tmrt = microclimate["Tmrt"].to_numpy()

    # Extrapolate wind to the reference heights with Eq. 2.
    v_1m1 = wind_at_height(v, 1.1, 1.1)          # PET reference (measurement height)
    v_10 = wind_at_height(v, 1.1, 10.0)          # UTCI reference (10 m)

    humidex = 0.10 * np.clip(rh - 30.0, 0.0, None)  # mild humidity load

    utci = ta + 0.30 * (tmrt - ta) - 1.10 * np.sqrt(v_10) + humidex
    pet = ta + 0.42 * (tmrt - ta) - 0.65 * np.sqrt(v_1m1) + 0.6 * humidex
    pmv = 0.45 * (ta - 26.0) + 0.11 * (tmrt - ta) - 0.35 * v_1m1 + 0.4 * humidex

    return pd.DataFrame({"UTCI_raw": utci, "PET_raw": pet, "PMV_raw": pmv})


def _calibrate_to_table2(raw: np.ndarray, index_name: str) -> np.ndarray:
    """Affine map raw index values so Table 2 class boundaries sit at the
    published quantiles (guaranteeing the published class frequencies)."""
    share1, bound1, share12, bound2 = _CALIBRATION[index_name]
    q1 = np.quantile(raw, share1)
    q2 = np.quantile(raw, share12)
    if abs(q2 - q1) < 1e-9:
        raise ValueError(f"degenerate calibration distribution for {index_name}")
    slope = (bound2 - bound1) / (q2 - q1)
    return bound1 + slope * (raw - q1)


def build_dataset(
    n_points: int = 1168,
    seed: int = SEED,
    microclimate_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generate the synthetic benchmark (or attach real microclimate data).

    Returns one row per measurement point with:
      - morphological features (orientation one-hot + continuous columns);
      - microclimate drivers (Ta, RH, V, Tmrt);
      - the three calibrated OTC indices (UTCI, PET, PMV).
    """
    rng = np.random.default_rng(seed)

    # Orientation: cover all Table 1 bins, overweighting the 45-degree classes
    # the paper's streets actually fall into.
    orientation_deg = rng.choice(
        [0.0, 30.0, 60.0, 100.0, 150.0, 180.0],
        size=n_points,
        p=[0.10, 0.15, 0.25, 0.20, 0.15, 0.15],
    )
    width = rng.uniform(*WIDTH_RANGE_M, size=n_points)
    hw = rng.uniform(*HW_RATIO_RANGE, size=n_points)
    floors = rng.integers(FLOORS_RANGE[0], FLOORS_RANGE[1] + 1, size=n_points).astype(float)
    trees = rng.integers(0, 40, size=n_points).astype(float)

    features = build_feature_frame(orientation_deg, width, hw, floors, trees)

    if microclimate_path is not None:
        microclimate = pd.read_csv(microclimate_path)
        missing = [c for c in MICROCLIMATE_COLUMNS if c not in microclimate.columns]
        if missing:
            raise ValueError(f"microclimate file missing columns: {missing}")
        if len(microclimate) != n_points:
            raise ValueError(
                f"microclimate file has {len(microclimate)} rows, expected {n_points}"
            )
        microclimate = microclimate[MICROCLIMATE_COLUMNS].reset_index(drop=True)
    else:
        microclimate = _microclimate_from_morphology(
            orientation_deg, width, hw, floors, trees, rng
        )

    raw = _approx_indices(microclimate)
    indices = pd.DataFrame(
        {
            index: _calibrate_to_table2(raw[f"{index}_raw"].to_numpy(), index)
            for index in INDEX_COLUMNS
        }
    )

    return pd.concat([features, microclimate, indices], axis=1)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load a user-supplied dataset CSV (e.g. the real field measurements).

    Required columns: the morphological features (or the raw fields
    orientation_deg / street_width_m / hw_ratio / n_floors / n_trees, which
    are encoded automatically) plus Ta, RH, V, Tmrt and at least one of
    UTCI / PET / PMV.
    """
    frame = pd.read_csv(csv_path)

    if "orientation_deg" in frame.columns and "orientation_45" not in frame.columns:
        encoded = build_feature_frame(
            frame["orientation_deg"],
            frame["street_width_m"],
            frame["hw_ratio"],
            frame["n_floors"],
            frame["n_trees"],
        )
        frame = pd.concat(
            [frame.drop(columns=["orientation_deg"]).reset_index(drop=True),
             encoded.reset_index(drop=True)],
            axis=1,
        )

    required = MICROCLIMATE_COLUMNS + ["orientation_45", "street_width_m", "hw_ratio"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"dataset CSV missing required columns: {missing}")
    if not any(index in frame.columns for index in INDEX_COLUMNS):
        raise ValueError(f"dataset CSV must contain at least one of {INDEX_COLUMNS}")
    return frame


def class_shares(labels: pd.Series) -> Dict[str, float]:
    """Fraction of each class in a label series (for QA against Table 2)."""
    counts = labels.value_counts(normalize=True)
    return {str(name): round(float(share), 4) for name, share in counts.items()}


def verify_table2_fractions(frame: pd.DataFrame, tolerance: float = 0.03) -> bool:
    """True if all nine class shares are within `tolerance` of Table 2."""
    from .features import classify_otc

    for index in INDEX_COLUMNS:
        labels = classify_otc(index, frame[index].to_numpy())
        shares = class_shares(pd.Series(labels))
        expected = CLASS_FRACTIONS[index]
        names = [entry[2] for entry in _ordered_defs(index)]
        for name, target in zip(names, expected):
            if abs(shares.get(name, 0.0) - target) > tolerance:
                return False
    return True


def _ordered_defs(index_name: str):
    from .config import OTC_CLASS_DEFS

    return OTC_CLASS_DEFS[index_name]
