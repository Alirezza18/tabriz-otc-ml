"""Test suite for the OTC-ML package.

Run with:
    python -m unittest discover -s tests -v

Covers:
- the paper's physical equations (Eqs. 1-2) against hand-computed values;
- Table 2 classification boundaries and class fractions;
- feature-frame construction and validation rules;
- the six estimator factories + balanced sample weighting;
- weighted-average evaluation metrics;
- a small end-to-end Bayesian-optimization smoke test.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from otc_ml.config import CLASS_FRACTIONS, CLASS_NAMES, FEATURES, OTC_CLASS_DEFS
from otc_ml.data import (
    build_dataset,
    mean_radiant_temp,
    verify_table2_fractions,
    wind_at_height,
)
from otc_ml.evaluate import evaluate_all_models, evaluate_classifier, format_results_table
from otc_ml.features import (
    build_feature_frame,
    classify_otc,
    class_label_to_int,
    one_hot_orientation,
    orientation_to_bin,
    validate_feature_frame,
)
from otc_ml.models import (
    ALGORITHMS,
    balanced_sample_weights,
    fit_estimator,
    make_estimator,
    predict_labels,
)


class TestPhysicsEquations(unittest.TestCase):
    def test_tmrt_zero_delta_equals_globe(self):
        # If globe == air temperature and V term vanishes via (Tg - Ta) = 0,
        # Tmrt must equal the globe temperature.
        self.assertAlmostEqual(mean_radiant_temp(30.0, 30.0, 1.0), 30.0, places=6)

    def test_tmrt_globe_above_air_radiatively_warmer(self):
        tmrt = mean_radiant_temp(30.0, 35.0, 1.0)
        self.assertGreater(tmrt, 35.0)

    def test_tmrt_convective_term_grows_with_wind(self):
        # The convective term 1.1e8*V^0.6/(eps D^0.4)*(Tg - Ta) enters the
        # fourth root with a POSITIVE sign when the globe is warmer than the
        # air, so a higher wind speed raises Tmrt for Tg > Ta (the delta
        # between globe and Tmrt shrinks toward the pure-radiative value in
        # relative terms only when Tg < Ta). Verify the paper-consistent
        # behavior directly: Tmrt increases with V when Tg > Ta.
        calm = mean_radiant_temp(30.0, 35.0, 0.5)
        windy = mean_radiant_temp(30.0, 35.0, 3.0)
        self.assertGreater(windy, calm)
        # And decreases with V when the globe is cooler than the air.
        calm_cool = mean_radiant_temp(35.0, 30.0, 0.5)
        windy_cool = mean_radiant_temp(35.0, 30.0, 3.0)
        self.assertLess(windy_cool, calm_cool)

    def test_wind_profile_scales_logarithmically(self):
        v10 = wind_at_height(2.0, measurement_height_m=1.1, target_height_m=10.0)
        expected = 2.0 * np.log(10.0 / 0.01) / np.log(1.1 / 0.01)
        self.assertAlmostEqual(v10, expected, places=6)

    def test_wind_profile_identity_at_same_height(self):
        self.assertAlmostEqual(
            wind_at_height(3.0, measurement_height_m=10.0, target_height_m=10.0),
            3.0,
            places=6,
        )


class TestTable2Classification(unittest.TestCase):
    def test_utci_boundaries(self):
        labels = classify_otc("UTCI", [26.0, 31.9, 32.0, 37.9, 38.0, 45.9])
        self.assertEqual(
            labels,
            [
                "Moderate heat stress",
                "Moderate heat stress",
                "Strong heat stress",
                "Strong heat stress",
                "Very strong heat stress",
                "Very strong heat stress",
            ],
        )

    def test_pet_boundaries(self):
        labels = classify_otc("PET", [29.0, 34.9, 35.0, 40.9, 41.0, 59.0])
        self.assertEqual(
            labels,
            [
                "Moderate heat stress",
                "Moderate heat stress",
                "Strong heat stress",
                "Strong heat stress",
                "Extreme heat stress",
                "Extreme heat stress",
            ],
        )

    def test_pmv_boundaries(self):
        labels = classify_otc("PMV", [2.0, 2.9, 3.0, 3.9, 4.0, 6.5])
        self.assertEqual(labels, ["Warm", "Warm", "Hot", "Hot", "Very hot", "Very hot"])

    def test_below_range_clips_to_first_class(self):
        self.assertEqual(classify_otc("UTCI", [-10.0]), ["Moderate heat stress"])

    def test_class_names_match_table2(self):
        self.assertEqual(
            CLASS_NAMES["UTCI"],
            ["Moderate heat stress", "Strong heat stress", "Very strong heat stress"],
        )
        self.assertEqual(
            CLASS_NAMES["PET"],
            ["Moderate heat stress", "Strong heat stress", "Extreme heat stress"],
        )
        self.assertEqual(CLASS_NAMES["PMV"], ["Warm", "Hot", "Very hot"])

    def test_label_int_encoding_is_ordered(self):
        labels = ["Very strong heat stress", "Moderate heat stress", "Strong heat stress"]
        self.assertEqual(class_label_to_int("UTCI", labels), [2, 0, 1])


class TestFeatureEngineering(unittest.TestCase):
    def test_orientation_binning(self):
        self.assertEqual(orientation_to_bin(30.0), "orientation_45")
        self.assertEqual(orientation_to_bin(60.0), "orientation_90")
        self.assertEqual(orientation_to_bin(100.0), "orientation_135")
        self.assertEqual(orientation_to_bin(150.0), "orientation_180")
        self.assertEqual(orientation_to_bin(180.0), "orientation_180")
        self.assertEqual(orientation_to_bin(0.0), "orientation_45")
    def test_one_hot_shape_and_values(self):
        frame = one_hot_orientation([0.0, 60.0, 150.0])
        self.assertEqual(list(frame.columns), FEATURES[:4])
        self.assertEqual(frame.sum(axis=1).tolist(), [1, 1, 1])
        self.assertEqual(frame["orientation_90"].tolist(), [0, 1, 0])

    def test_build_feature_frame_layout(self):
        frame = build_feature_frame([90.0], [12.0], [1.5], [4.0], [10.0])
        self.assertEqual(list(frame.columns), FEATURES)
        self.assertEqual(frame["orientation_135"].iloc[0], 1)
        self.assertEqual(frame["street_width_m"].iloc[0], 12.0)

    def test_validation_rejects_bad_frames(self):
        good = build_feature_frame([90.0], [12.0], [1.5], [4.0], [10.0])
        validate_feature_frame(good)  # must not raise

        two_active = good.copy()
        two_active.loc[0, "orientation_45"] = 1
        with self.assertRaises(ValueError):
            validate_feature_frame(two_active)

        bad_width = build_feature_frame([90.0], [99.0], [1.5], [4.0], [10.0])
        with self.assertRaises(ValueError):
            validate_feature_frame(bad_width)

        nan_frame = build_feature_frame([90.0], [np.nan], [1.5], [4.0], [10.0])
        with self.assertRaises(ValueError):
            validate_feature_frame(nan_frame)


class TestSyntheticDataset(unittest.TestCase):
    def test_dataset_shape_and_columns(self):
        frame = build_dataset(n_points=300, seed=1)
        for column in FEATURES + ["Ta", "RH", "V", "Tmrt", "UTCI", "PET", "PMV"]:
            self.assertIn(column, frame.columns)
        self.assertEqual(len(frame), 300)

    def test_class_fractions_match_table2(self):
        frame = build_dataset(n_points=1168, seed=42)
        self.assertTrue(verify_table2_fractions(frame, tolerance=0.03))

    def test_reproducible_with_seed(self):
        a = build_dataset(n_points=100, seed=7)
        b = build_dataset(n_points=100, seed=7)
        pd.testing.assert_frame_equal(a, b)


class TestModels(unittest.TestCase):
    def test_all_six_algorithms_constructible(self):
        for algorithm in ALGORITHMS:
            estimator = make_estimator(algorithm, {})
            self.assertIsNotNone(estimator)

    def test_unknown_algorithm_raises(self):
        with self.assertRaises(ValueError):
            make_estimator("LGBM", {})

    def test_balanced_weights_upsample_minority(self):
        y = np.array([0] * 90 + [1] * 10)
        weights = balanced_sample_weights(y)
        self.assertGreater(weights[y == 1].mean(), weights[y == 0].mean())

    def test_fit_estimator_learns_separable_problem(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, len(FEATURES)))
        y = (X[:, 4] > 0).astype(int)  # width separates classes
        for algorithm in ALGORITHMS:
            estimator = fit_estimator(algorithm, {}, X, y)
            accuracy = (predict_labels(estimator, X, y) == y).mean()
            self.assertGreater(accuracy, 0.8, f"{algorithm} failed to fit")

    def test_catboost_predict_returns_flat_labels(self):
        # Regression guard: CatBoost.predict() returns an (n, 1) column, so a
        # naive `pred == y` broadcast corrupts every accuracy computation
        # (200 hits out of a 200x200 matrix -> exactly 0.5002).
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, len(FEATURES)))
        y = (X[:, 4] > 0).astype(int)
        estimator = fit_estimator("CatBoost", {}, X, y)
        predictions = predict_labels(estimator, X, y)
        self.assertEqual(predictions.shape, (120,))
        self.assertGreater((predictions == y).mean(), 0.8)


class TestEvaluation(unittest.TestCase):
    def test_perfect_predictions_score_100(self):
        y = [0, 1, 2, 0, 1, 2]
        metrics = evaluate_classifier(y, y)
        self.assertEqual(metrics["accuracy"], 100.0)
        self.assertEqual(metrics["f1"], 100.0)

    def test_weighted_average_handles_imbalance(self):
        y_true = [0] * 80 + [1] * 20
        y_pred = [0] * 80 + [1] * 10 + [0] * 10
        metrics = evaluate_classifier(y_true, y_pred)
        self.assertAlmostEqual(metrics["accuracy"], 90.0, places=1)

    def test_format_table_renders_rows(self):
        rows = [{"algorithm": "RF", "accuracy": 76, "precision": 75, "recall": 76, "f1": 75}]
        table = format_results_table(rows, "PET")
        self.assertIn("RF", table)
        self.assertIn("PET", table)


class TestBayesianOptimization(unittest.TestCase):
    def test_bo_smoke_test_improves_over_worst(self):
        from otc_ml.bayes_opt import optimize_hyperparameters

        rng = np.random.default_rng(0)
        X = rng.normal(size=(240, len(FEATURES)))
        y = (X[:, 4] + 0.5 * X[:, 5] > 0).astype(int)

        best_params = optimize_hyperparameters("DT", X, y, n_trials=4, seed=0)
        self.assertIsInstance(best_params, dict)
        self.assertIn("max_depth", best_params)

        estimator = fit_estimator("DT", best_params, X, y)
        accuracy = (predict_labels(estimator, X, y) == y).mean()
        self.assertGreater(accuracy, 0.7)


class TestEndToEnd(unittest.TestCase):
    def test_single_index_fast_pipeline(self):
        from otc_ml.pipeline import run_single_index

        dataset = build_dataset(n_points=360, seed=3)
        outcome = run_single_index(
            dataset, "PMV", n_trials=2, algorithms=["DT", "RF"], seed=3
        )
        self.assertIn(outcome["best_algorithm"], {"DT", "RF"})
        self.assertEqual(len(outcome["results"]), 2)
        for row in outcome["results"]:
            for metric in ("accuracy", "precision", "recall", "f1"):
                self.assertGreaterEqual(row[metric], 0.0)
                self.assertLessEqual(row[metric], 100.0)


if __name__ == "__main__":
    unittest.main()
