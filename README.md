# stock_analysis

## Synthetic Pipeline

- Entry point: `main.py`
- Uses synthetic data generation and existing evaluation flow.
- NN naming is unified to `NN`.
- Legacy alternative NN path has been removed.

### Run

```bash
python main.py
```

### Outputs

- `data/output/<run_id>/<run_id>_oof_results.csv`
- `data/output/<run_id>/<run_id>_metrics_summary.csv`
- gain/calibration/SHAP plots in the same output directory

## Marketing Pipeline

- Entry point: `marketing_main.py`
- Input is real marketing data CSV loaded with `pd.read_csv(..., sep=";")`.
- Models: `LGBM` / `Logistic` / `NN` / `EDL` (optional with `--enable-edl`)
- NN implementation is only `src/models/nn.py` (`NeuralNetModel`).
- Cross-validation: `StratifiedKFold`
- Preprocessing is fold-local to avoid leakage:
  - missing value imputation
  - one-hot encoding for categorical columns

### Data Placement

`bank-full.csv` is not tracked by Git due to `.gitignore` (`*.csv`).

Place the file at:

- `data/input/bank-full.csv`

### Run

```bash
python marketing_main.py --data-path data/input/bank-full.csv
```

Optional debug columns for call list:

```bash
python marketing_main.py --data-path data/input/bank-full.csv --debug-score-columns
```

Enable EDL:

```bash
python marketing_main.py --data-path data/input/bank-full.csv --enable-edl
```

Enable EDL with uncertainty-weighted score:

```bash
python marketing_main.py --data-path data/input/bank-full.csv --enable-edl --score-mode edl_uncertainty_weighted
```

### Output Column Contract

`oof_results.csv`

- `y_true`
- `y_pred_proba_lgbm`
- `y_pred_proba_logit`
- `y_pred_proba_nn`
- optional EDL columns (when `--enable-edl`):
  - `y_pred_proba_edl`
  - `y_pred_uncertainty_edl`

`metrics_summary.csv`

- `model`
- `auc`
- `ap`

`call_list_top200.csv`

- required: `row_id`, `score`, `rank`
- optional (debug flag): `score_components`, `n_models_used`

### Scoring Layer Design

Score construction is centralized in `src/marketing/scoring.py`.

- API: `build_call_score(oof_df, schema) -> pd.Series`
- Current logic: mean of available probability columns defined by schema.
- Future EDL integration should be added only in this layer, where uncertainty-aware scoring can be implemented.

Schema constants are defined in `src/marketing/schemas.py`:

- `PROBA_PREFIX = "y_pred_proba_"`
- `UNC_PREFIX = "y_pred_uncertainty_"`
- `REQUIRED_MODELS = ["lgbm", "logit", "nn"]`
- `OPTIONAL_MODELS = ["edl"]`
