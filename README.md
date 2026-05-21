# Kaggle-Pipeline

A config-driven AutoML pipeline for tabular Kaggle competitions. Point it at a
competition, set a handful of metaparameters, and it preprocesses the data, runs
an adaptive randomized search across a zoo of models while keeping a
leaderboard, and stacks the winners into an ensemble that writes a submission.

It is designed for the **"thin notebook on Kaggle"** workflow: the heavy logic
lives in this installable package, and a small notebook installs it from GitHub,
sets a few parameters, and runs.

## How it works

Training and exploration are two independent flows. Training (`run`) never
imports plotting libraries or renders anything:

```
run:      detect env → load data → preprocess + build context
                     → adaptive model search (leaderboard) → stacked ensemble → submission.csv

analyze:  detect env → load data → EDA (metadata, correlations, pairwise plots)
```

- **Preprocessing** (`preprocessing/`): feature engineering from `pandas.eval`
  expressions, categorical ordering/typing, and ordinal encoding of detected
  ordinal columns.
- **Search** (`search/`): each *step* draws a batch of model classes from the
  leaderboard, samples hyperparameters (scaled by a per-class `complexity`
  knob), cross-validates them in parallel, and records out-of-fold predictions.
  The leaderboard keeps the best per class under capacity bounds and adapts each
  class's complexity based on score-per-log-compute-time.
- **Ensembling** (`search/judge.py`): the selected models' out-of-fold
  predictions are stacked with a logistic-regression meta-model
  (`RandomizedSearchCV`), and the decoded test predictions become the submission.

The whole run is checkpointed to disk after every step, so an interrupted Kaggle
kernel resumes from the last saved leaderboard.

## Project layout

```
src/kaggle_pipeline/
├── config/         # Config dataclass, YAML loader, Kaggle/Colab/local env detection
├── data/           # CSV loading (+ optional speed-up subsampling)
├── preprocessing/  # transformers, column helpers, target transforms, pretrain pipeline
├── context/        # PipelineContext: fitted run-wide state threaded everywhere
├── scoring/        # scoring-metric resolution
├── eda/            # exploratory plots & reports (used only by analyze)
├── models/         # Model base class, registry, one file per model in definitions/
├── search/         # cross-validation, leaderboard, judge (search + ensemble)
├── training/       # the step loop under a time budget
├── submission/     # writes the submission CSV
├── pipeline.py     # run() — the training entry point
├── analysis.py     # analyze() — the standalone EDA entry point
└── cli.py          # `kaggle-pipeline run|analyze --config ...`
configs/            # example YAML config (Playground Series S6E4)
notebooks/          # kaggle_runner.ipynb — the thin notebook to upload to Kaggle
tests/              # unit + end-to-end smoke tests
```

## Quickstart on Kaggle

Upload [`notebooks/kaggle_runner.ipynb`](notebooks/kaggle_runner.ipynb), or paste
these cells. Enable **Internet** in the notebook settings.

```python
# 1. Install (pin a tag for reproducibility, e.g. @v0.1.0)
!pip install -q git+https://github.com/HkHk2Prod/Kaggle-Pipeline.git

# 2. Configure — the few things that change per competition
from kaggle_pipeline import Config
cfg = Config(
    competition="playground-series-s6e4",
    target="Irrigation_Need",
    scoring="balanced_accuracy",
    prediction_aim="category",
    feature_expressions=["soil_lt_25 = Soil_Moisture < 25"],
)

# 3. Run
from kaggle_pipeline import run
run(cfg)  # writes submission.csv to /kaggle/working
```

For **code competitions with no internet at scoring time**, add the repo as a
Kaggle Dataset and `sys.path` it instead of `pip install` — see the fallback
section in the runner notebook.

## Local / CLI usage

```bash
# Install (uv recommended)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

# Train from a YAML config (set data_dir to your local data folder)
.venv/bin/kaggle-pipeline run --config configs/playground-s6e4.yaml

# Or explore the data first (separate, no training)
.venv/bin/kaggle-pipeline analyze --config configs/playground-s6e4.yaml
```

Or from Python:

```python
from kaggle_pipeline import Config, run, analyze

cfg = Config.from_yaml("configs/playground-s6e4.yaml")
analyze(cfg)   # optional: render EDA only
run(cfg)       # train + write submission
```

## Configuration

All knobs live in one `Config` object (see [`configs/playground-s6e4.yaml`](configs/playground-s6e4.yaml)
for a documented example). The most important fields:

| Field | Meaning |
| --- | --- |
| `competition` | Used to derive default Kaggle/Colab data paths |
| `target`, `id_col` | Target and id column name(s) |
| `task` | `classification` (the only implemented path). `regression` raises `NotImplementedError` — see [Regression](#regression). |
| `scoring` | `balanced_accuracy`, `roc_auc` or `neg_root_mean_squared_error` |
| `prediction_aim` | `category` or `probability` |
| `feature_expressions` | `pandas.eval` expressions for new features |
| `categorical_encoding` | Per-column encoding for non-native models (see below) |
| `n_steps`, `num_models`, `step_batch_size` | Search budget and leaderboard size |
| `ensemble_length`, `ensemble_min_repr` | Ensemble size and min models kept per class |
| `speed_up` | Subsample data for fast debugging |
| `max_running_time` | Stop the loop before exceeding this many seconds |
| `train_csv`, `test_csv`, `sample_csv` | CSV filenames within `data_dir` |
| `data_dir`, `storage_dir` | Override paths (required `data_dir` when running locally) |

### Autodetection

`competition` is the only field you must set. Leave `target`, `task`,
`scoring`, `prediction_aim` or the CSV filenames unset (or `None`/`null`) and
they are inferred from the data when it loads — the chosen value is printed as a
`[autodetect] ...` line so the run is reproducible from its log:

- **CSV filenames** — the first file in `data_dir` whose name contains
  `train` / `test` / `sample`.
- **`target`** — the last non-id column of the train frame.
- **`task`** — `classification` for text or low-cardinality integer targets,
  `regression` for continuous numeric ones. A `regression` result (autodetected
  or set explicitly) raises `NotImplementedError` — see [Regression](#regression).
- **`prediction_aim`** — `probability` for classification.
- **`scoring`** — `roc_auc` for binary targets, `balanced_accuracy` for
  multiclass, `neg_root_mean_squared_error` for regression.

So a minimal config can be as short as `Config(competition="…", data_dir="…")`.

### Categorical encoding

High-cardinality categoricals (e.g. a `driver` column with dozens of names) are
handled per model by capability:

- **Models that handle categoricals natively** — CatBoost, XGBoost, LightGBM and
  HistGB — are handed the **raw column**. *Capability wins:* `categorical_encoding`
  is ignored for them. (Exception: HistGB encodes any column above its native
  255-level cap, since sklearn cannot use more than that natively.)
- **Models that can't** — RandomForest and LogisticRegression — get the column
  **encoded** per `categorical_encoding`, defaulting to **frequency encoding**.
  This replaces one-hot, so a high-cardinality column becomes a single numeric
  feature instead of one dummy per level (and unseen test levels no longer error).

Set strategies per column; unset columns default to `frequency`:

```yaml
categorical_encoding:
  driver: frequency   # frequency | target | onehot | ordinal | native | drop
  region: onehot
```

The resolved plan is printed as `[encoding] ...` lines when the data loads, so
each column's chosen strategy is visible in the run log.

## Adding a model

Drop a module in `src/kaggle_pipeline/models/definitions/` with a `Model`
subclass decorated with `@register_model`, implementing `generate_distribution`
(hyperparameter distribution, scaled by `complexity`) and `build_pipeline`
(the sklearn pipeline). Import it in `definitions/__init__.py`. It then
participates in the search automatically.

## Behaviour changes vs. the original notebook

The search/leaderboard/ensemble logic is preserved; a few correctness fixes were
applied while packaging:

- **Ordinal `detect_ordinal_order_cols`**: the notebook had two definitions; the
  later one shadowed the correct case-insensitive version with a case-sensitive
  one. The case-insensitive version is kept.
- **Ordinal encoding stays numeric**: `pandas >= 3.0` keeps the `category` dtype
  through `.map`, so ordinal-encoded columns are now explicitly coerced numeric
  (matching the behaviour on Kaggle's pandas 2.x).
- **Reloaded ensemble models use their tuned params**: the notebook rebuilt a
  loaded model's pipeline with random hyperparameters and never re-applied the
  saved ones, so ensemble members were refit incorrectly. `Model.load` now
  rebuilds the pipeline from the saved parameters.
- **Task type is explicit or opt-in autodetected**: set `task` directly, or
  leave it unset for a deliberate, announced dtype-based inference — replacing
  the notebook's implicit, silent (and buggy) dtype sniffing.

## Development

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q          # tests (incl. an end-to-end smoke run)
.venv/bin/python -m pytest --cov=kaggle_pipeline --cov-report=term-missing  # coverage
.venv/bin/ruff check src tests         # lint
.venv/bin/ruff format src tests        # format
.venv/bin/mypy                         # type-check (config in pyproject.toml)
```

## Regression

**Regression is not implemented yet.** Only classification is wired end-to-end:
there are no regressor model definitions and the ensembling/prediction path only
handles classification. To avoid a confusing late failure, a `regression` task
fails fast with `NotImplementedError` — both when you set `task: regression`
explicitly and when it is autodetected from a continuous numeric target. For now,
keep targets categorical (or set `task: classification`). Regression support is
planned.

## Dependencies

Installing the package pulls in everything below (declared in
[`pyproject.toml`](pyproject.toml)); there are no optional groups yet, so the
plotting libraries are installed even though `run` never imports them.

| Dependency | Used by |
| --- | --- |
| `numpy`, `pandas`, `scipy` | everywhere — data frames, arrays, distributions |
| `scikit-learn` (>=1.4) | preprocessing, CV, encoders, the stacking meta-model |
| `lightgbm`, `catboost`, `xgboost` | the gradient-boosting models in the search |
| `joblib` | parallel cross-validation across the model batch |
| `pyyaml` | loading a `Config` from a YAML file |
| `matplotlib`, `seaborn` | **`analyze` (EDA) only** — never imported by `run` |

So a pure-training run (`run`) needs everything except `matplotlib`/`seaborn`;
those are pulled in only for the standalone `analyze` flow. Dev extras
(`pip install -e ".[dev]"`) add `pytest` and `ruff`.

## Notes

- The original exploratory notebook is preserved untouched as
  [`Kaggle_Pipeline.ipynb`](Kaggle_Pipeline.ipynb) for reference.
- Runs are reproducible by default: `Config.seed` defaults to `42` and threads
  through the model search, the leaderboard's class selection and the ensemble
  search. Set `seed=None` for non-reproducible behaviour (the notebook default).
- Progress output goes through Python's `logging` (the `kaggle_pipeline` logger)
  rather than `print`; the entry points configure it at `INFO` so it shows by
  default. Adjust the logger's level to quiet or redirect it.

## License

MIT — see [LICENSE](LICENSE).
