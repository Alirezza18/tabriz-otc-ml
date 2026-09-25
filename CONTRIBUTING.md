# Contributing to TabrizOTC-ML

Thanks for helping improve the reference implementation of the
Alinasab et al. (2025) OTC-ML pipeline! This guide gets you from clone to
merged PR in a few minutes.

## Development setup

```bash
git clone https://github.com/Alirezza18/tabriz-otc-ml.git
cd tabriz-otc-ml
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Or, if you prefer Docker (nothing to install except Docker itself):

```bash
docker compose run --rm study --n-trials 3 --out results/
```

## Running the tests

The full suite (28 tests: physics equations, Table 2 boundaries, features,
models, Bayesian optimization, end-to-end) must pass before any merge:

```bash
python -m unittest discover -s tests -v
```

CI runs the same suite plus a Docker build + in-container study smoke test on
every pull request — if it's green locally, CI should pass too.

## How we work (branch protection is ON)

1. **Never push directly to `main`** — it's protected and will be rejected.
2. Create a feature branch: `git checkout -b feature/your-change`
3. Commit with clear messages (what + why).
4. Open a **pull request** against `main`, describing what changed and why.
5. CI must be green; then merge (squash or merge commit both fine).

## Ground rules

- **Reproducibility first**: every random draw must route through a seeded
  `Generator`; new features need a `seed` parameter and a test.
- **Paper traceability**: constants trace back to the paper (Tables 1–3,
  Eqs. 1–2) — cite the table/equation in comments when adding or changing one.
- **Synthetic data honesty**: the shipped benchmark is explicitly synthetic;
  don't add anything that blurs that line (no fabricated "measured" values).
- **Keep dependencies lean** in `requirements.txt`; matplotlib is deliberately
  excluded for now (outputs are CSVs).
- Style: the Pylint CI job is **errors-only and non-blocking** — real bugs
  matter, formatting nitpicks don't.

## Reporting problems

Open an issue with: what you ran (command + `--seed`), what you expected,
what happened. If it's data-related, include the `verify_table2_fractions`
output.

## Data policy

The Tabriz field dataset is available **from the corresponding author on
request** and must never be committed to this repository (`.gitignore`
already excludes `data/*.csv`).

## Maintainers

- **Alireza Karimi** ([@Alirezza18](https://github.com/Alirezza18)) — code author
- **Niloufar Alinasab** ([@nfaralinasab-hub](https://github.com/nfaralinasab-hub)) — field-study lead & corresponding author
