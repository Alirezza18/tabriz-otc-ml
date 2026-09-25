#!/usr/bin/env python
"""Run the full study from the command line.

Examples:
    # Quick demonstration run on the synthetic benchmark (fast):
    python scripts/run_study.py --n-trials 5 --out results/

    # Paper-faithful run (60 BO trials x 6 algorithms x 3 indices):
    python scripts/run_study.py --out results/

    # With your own field dataset:
    python scripts/run_study.py --csv data/my_measurements.csv --out results/

Outputs (under --out):
    results_<index>.csv   one Table-3-style metrics row per algorithm
    best_params_<index>.json
    shap_importance_<index>.csv   (Fig. 3 equivalent)
    shap_directions_<index>.csv   (Fig. 4 equivalent)
    summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running straight from a clone without `pip install .`:
#   python scripts/run_study.py ...
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from otc_ml.pipeline import run_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OTC-ML study")
    parser.add_argument("--csv", default=None, help="Path to a field-measurement CSV")
    parser.add_argument("--n-trials", type=int, default=60, help="BO trials per algorithm")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--no-shap", action="store_true", help="Skip the SHAP stage")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    study = run_study(
        csv_path=args.csv,
        n_trials=args.n_trials,
        seed=args.seed,
        with_shap=not args.no_shap,
        verbose=True,
    )

    summary = {"best_models": study["best_models"], "expected_best_models": study["expected_best_models"]}

    for index_name, outcome in study["indices"].items():
        import pandas as pd

        pd.DataFrame(outcome["results"]).to_csv(out_dir / f"results_{index_name}.csv", index=False)
        (out_dir / f"best_params_{index_name}.json").write_text(
            json.dumps(outcome["best_params"], indent=2, default=str)
        )
        summary[index_name] = {"best": outcome["best_algorithm"], "results": outcome["results"]}

    for index_name, shap_out in study.get("shap", {}).items():
        import pandas as pd

        shap_out["per_class_importance"].to_csv(
            out_dir / f"shap_importance_{index_name}.csv", index=False
        )
        shap_out["binary_directions"].to_csv(
            out_dir / f"shap_directions_{index_name}.csv", index=False
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n==== RESULTS (Table 3 equivalents) ====")
    for index_name, table in study["tables"].items():
        print(table)
    print("\nBest models found:", study["best_models"])
    print("Paper's best models:", study["expected_best_models"])
    print(f"\nAll outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
