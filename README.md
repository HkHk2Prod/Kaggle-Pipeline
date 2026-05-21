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
| `task` | `classification` or `regression` (classification is the working path) |
| `scoring` | `balanced_accuracy` or `roc_auc` |
| `prediction_aim` | `category` or `probability` |
| `feature_expressions` | `pandas.eval` expressions for new features |
| `n_steps`, `num_models`, `step_batch_size` | Search budget and leaderboard size |
| `ensemble_length`, `ensemble_min_repr` | Ensemble size and min models kept per class |
| `speed_up` | Subsample data for fast debugging |
| `max_running_time` | Stop the loop before exceeding this many seconds |
| `data_dir`, `storage_dir` | Override paths (required `data_dir` when running locally) |

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
- **Task type is explicit** (`task` config) instead of an implicit, fragile
  dtype check on the target.

## Development

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q          # tests (incl. an end-to-end smoke run)
.venv/bin/ruff check src tests         # lint
.venv/bin/ruff format src tests        # format
```

## Notes

- The original exploratory notebook is preserved untouched as
  [`Kaggle_Pipeline.ipynb`](Kaggle_Pipeline.ipynb) for reference.
- Regression (`task: regression`) is stubbed but not yet implemented end-to-end,
  matching the original notebook's state.

## License

MIT — see [LICENSE](LICENSE).
