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

analyze:  detect env → load data → EDA (metadata, correlations, pairwise plots)  [opt-in: run_eda]
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

## Evolutionary architecture

> **Status: in progress.** This section is the *design contract* for an
> architectural rework that lives alongside the v1 search above, under
> [`src/kaggle_pipeline/evolution/`](src/kaggle_pipeline/evolution/). The v1
> `run`/`analyze` flow continues to work unchanged; the evolutionary layer
> reuses v1's model registry, cross-validation, scoring and `PipelineContext`
> rather than replacing them. Where integration is incomplete the code carries
> explicit `TODO`s and safe stubs.

### Philosophy

The pipeline treats feature engineering and model fitting as an **evolutionary
search problem**. Features are generated, scored, selected, reused, and deleted
over time. Models are represented as *genomes* composed of dependent *genes*. A
gene can represent a selected feature reference, an encoding choice, a model
parameter, or a resource setting. **Mutations create child models from parent
models** — the parent is never altered in place. After training, the child model
is compared with its parent, and the system records which mutated genes improved
the score, changed behaviour, or increased compute cost.

This is an *evolutionary* search but deliberately **not** a blind random genetic
search. Selection is score-aware, cost-aware, feature-aware, and
mutation-history-aware. The system does not try to maximize validation score at
any cost: it optimizes a **cost-aware utility** that accounts for validation
score, score stability, compute time, and (eventually) model diversity and
ensemble usefulness.

The single most important decision behind the design:

- **Features are global.** Feature *usage* is model-specific.
- **Model mutation produces child models.** Feature mutation produces child features.
- **Evaluation combines intrinsic feature scores with downstream model impact.**

### Logical features vs. physical columns

The design separates four concepts that the v1 pipeline blurred together:

| Concept | What it is |
| --- | --- |
| **`OriginalFeature`** | A raw column from the input dataset. |
| **`FeatureGenome`** | A *global, logical* feature definition — a **recipe**, e.g. `price`, `log(price)`, `price / area`, `city + product_type`. Immutable once created. |
| **`FeatureMaterialization`** | The actual computed values of a feature in a specific *context* (a fold, a sample, train vs. test). |
| **`MaterializedColumn`** | A physical column (or block of columns) handed to a model. |

A single logical `FeatureGenome` can expand to many materialized columns (e.g.
one-hot of a categorical), and is materialized differently per context (global
train, `fold_0_valid`, an out-of-fold encoding scheme, a fixed evaluation
sample). `max_active_features` caps *logical* features; a planned
`max_materialized_columns` caps the *physical* width.

### Global feature registry

Generated features are represented as global `FeatureGenome` objects rather than
private genes inside individual model genomes. A `FeatureGenome` defines the
recipe for creating a feature: its source features, transformation operator,
parameters, output type, and lineage. **Model genomes do not own feature
definitions.** Instead they contain `FeatureReferenceGene` objects that point to
feature IDs in the global `FeatureRegistry`. This lets the same generated feature
be reused by many models, scored once globally, correlated against other
features, and tracked across the entire search.

The `FeatureRegistry` is the source of truth for features. It stores all original
and generated genomes; tracks which are active, protected, and their scores,
usage and similarity stats; rejects duplicate recipes by hash; enforces
`max_active_features` (originals excepted); materializes features on demand; and
hands out selection probabilities and candidate parents. It never trains models —
it delegates to a `FeatureEvaluator`/`FeatureMaterializer`.

### Why features are global, not private model genes

If correlation and redundancy were computed inside private model objects, the
same generated feature appearing in ten models would be scored ten times from ten
incomplete views, and feature-level information (target correlation, redundancy
against the active pool, tree importance, drift, cost, downstream usage) would be
lost every time a model that used it was pruned. Computing feature evaluation
**at the registry level** keeps one authoritative score per feature and makes
cross-feature correlation meaningful. Encodings, by contrast, *are*
model-specific and live as genes inside model genomes (see below).

### Feature recipes, hashes, and why names are not enough

A human-readable name like `mul__income__age__944c10` is convenient in logs,
CSVs, Parquet and model output — but it is **not** the source of truth. Two
features with subtly different parameters could share a base name; very long
generated names are unusable in practice. The source of truth is the
**canonical `FeatureRecipe`** and its hash:

```
FeatureRecipe(
  transform_name, parent_feature_ids, parameters,
  output_type, version, uses_target, requires_oof,
  fold_context (only if target-dependent), metadata
)
```

Hashing rules:

1. Hash the **canonical recipe**, never the human name.
2. Sort parameter keys before hashing.
3. Include a transform **version** so behaviour changes invalidate old hashes.
4. Include parent feature IDs (or parent recipe hashes) consistently.
5. **Commutative** operators (add, multiply, category-combine) canonicalize
   parent order before hashing; **non-commutative** operators (subtract, divide)
   preserve order.

If two generated features share a canonical recipe they share a `feature_id`
(the duplicate is rejected/reused). Generated names use `readable_name +
short_hash`, e.g. `log__price__a83f91`, `div__price__area__9fa21c`,
`catjoin__city__product_type__cad882`, `freq__merchant_id__df31aa`.

### Feature materialization

A `FeatureGenome` is a recipe; a `FeatureMaterialization` is data. A
materialization records `feature_id`, `data_version_id`, `context_id`, optional
`fold_id`/`sample_id`, `values_hash`, `materialized_width`, `dtype`, `n_rows`, a
memory estimate, and an optional cache location. **Context matters**:
`global_train`, `global_test`, `fold_0_train`, `fold_0_valid`,
`oof_encoding_fold_scheme_1`, `feature_eval_sample_v1`. Target-dependent features
(target encoding, target means by group) must **never** be materialized globally
in a leaky way — their evaluation materialization is out-of-fold. Such recipes
are flagged `uses_target=True` and `requires_oof=True`.

### Feature transformations

Transformations are a class hierarchy under a `FeatureTransformation` base that
declares `name`, `input_types`, `output_type`, `arity`, `is_commutative`,
`uses_target`, `requires_oof`, `default_parameters`, `parameter_space`,
`cost_estimate`, and implements `validate_inputs`, `sample_parameters`, `apply`,
`generate_recipe`, `generate_name`, `validate_output`. Every transform validates
input types, rejects constant / near-constant outputs and excessive NaN/Inf,
handles Inf safely, and tracks generation cost and failure stats.

Initial numeric transforms: identity (originals), log/log1p, sqrt, square, rank,
zscore/standardize, min-max, clip/winsorize, binning, missing-indicator,
add, subtract, multiply, safe-divide, absolute-difference, ratio `x/(x+y+ε)`,
min, max. Initial categorical transforms: category-combination (with another
categorical or a binned numeric), frequency encoding, count encoding,
rare-category grouping, hash encoding, one-hot (as an *encoding*, not a base
feature). Target encoding is **planned** (stubbed) and, when added, must be
`uses_target=True`, `requires_oof=True`.

### Feature generation

A `FeatureGenerator` produces new `FeatureGenome` objects by: sampling one or
more parent features from the registry using feature probabilities; sampling a
transformation and its parameters; building a `FeatureRecipe`; checking the
recipe hash for duplicates; materializing/evaluating the candidate; and inserting
it into the registry only if it is useful enough. Candidates per batch default to
`ceil(max_active_features * feature_generation_ratio)` = `ceil(300 * 0.10)` = 30.

Initial rule: **generated features may only use original features as parents**
(`allow_generated_feature_parents = False`, `max_feature_depth = 1`). The design
already carries `depth`/`complexity` so deeper composition can be enabled later
with depth and complexity penalties.

### Feature scoring and credit assignment

Feature scores live in an extensible `FeatureScoreSet` — a mapping of named
`Score` objects, each with `value`, `higher_is_better`, `weight`, an optional
`normalized_value`, and metadata — so new scores are added without reshaping a
rigid struct. Initial intrinsic scores: `target_correlation`, `redundancy`
(negative), `tree_importance`. Reserved for later: missingness, drift, stability,
generation cost, materialized width, complexity, and the downstream scores below.

Each feature earns both an **intrinsic** score and **downstream model credit**,
combined with **confidence weighting** so early search leans on intrinsic signal
and later search leans on observed downstream impact:

```
feature_intrinsic   = w1·target_corr − w2·redundancy + w3·tree_importance
                      − w4·complexity − w5·cost
feature_downstream  = a1·avg_add_delta − a2·avg_remove_delta
                      + a3·avg_usage_credit + a4·elite_usage_rate
beta  = n_downstream_obs / (n_downstream_obs + k)   # confidence in downstream
alpha = 1 − beta
feature_utility = alpha·intrinsic + beta·downstream + original_feature_bonus
```

The cleanest downstream signal is **add/remove mutation credit**: if a child that
*added* feature `D` beats its parent, `D` gets positive add-credit; if removing
`D` improves a child, `D` gets remove-credit. Model-usage credit (optionally
weighted by per-model feature importance) is noisier and labelled as such.

Utility is turned into a selection probability with softmax plus an exploration
floor:

```
p(feature_i) = (1 − ε)·softmax(utility_i) + ε·uniform
```

Scores are rank-/robust-normalized before combining.

### Feature similarity and correlation

Because features are global, redundancy is computed between global feature
*materializations*, never inside private model genes. A `FeatureSimilarity`
component maintains **sparse top-k** similarity (not a forever-dense matrix),
computed on a fixed **reference evaluation sample** (`feature_eval_sample_v1`) so
correlations and value/sample fingerprints are comparable. Each feature can carry
a fingerprint: `recipe_hash`, `value_hash_on_sample`, sample signature,
distribution signature, missing rate, numeric mean/std/quantiles, categorical
cardinality/top categories. Numeric similarity ships first; categorical
similarity is planned.

### Feature deletion

The active pool is capped (`max_active_features = 300`). **Original features are
protected** and are never deletion candidates; if there are more originals than
the cap, the effective active limit rises to the original count. Generated
features can be deactivated but **remain reproducible** through their recipe. New
features get a creation **cooldown** before they are eligible for eviction. When
the pool is full and a new generated feature beats the weakest *removable*
feature, the weakest is deactivated and the new one activated. Deletion ranks on
more than intrinsic utility:

```
deletion_score = feature_utility + active_model_usage_bonus
               + elite_model_usage_bonus − redundancy_penalty − cost_penalty
```

### Model genomes and the gene system

A `ModelGenome` is immutable once created/trained; mutation creates a child. It
holds `model_id`, optional `parent_model_id`, a `BaseModelGene` (the model
family — **immutable within a genome**; changing family is a new genome, not a
mutation), `FeatureReferenceGene`s, `ParameterGene`s, `ResourceGene`s, metadata,
a `genome_hash`, `created_at_batch`, `mutation_history`, `status`, and optional
score/utility.

```
ModelGenome
  BaseModelGene(LightGBM)
  FeatureReferenceGene(f_price)
  FeatureReferenceGene(f_area)
  FeatureReferenceGene(f_div_price_area)
  ParameterGene(learning_rate = 0.03)
  ParameterGene(num_leaves = 64)
  ResourceGene(fidelity_level = 1)
```

All genes derive from a base `Gene` (`gene_id`, `gene_type`, `value`, `mutable`,
optional `parent_gene_id`, `child_gene_ids`, metadata, `mutation_stats`) with
`validate`, `copy`, `mutate(signed_amount, context) -> Gene | list[Gene]`,
`to_serializable`, `hash_component`. Subclasses: `BaseModelGene`,
`FeatureReferenceGene` (whose children may be `EncodingGene`s), `EncodingGene`,
`ParameterGene`, `ResourceGene`.

### Encodings are model-specific

Encoding is *not* baked into a `FeatureGenome`. The logical feature `city` can be
native-categorical in CatBoost, one-hot in Ridge, and count-encoded in LightGBM
— so encoding is an `EncodingGene` that is a **child of a `FeatureReferenceGene`**
inside each model genome. Remove the feature reference and its encoding child is
removed with it.

### Parameter genes and mutation amounts

`mutate(signed_amount)` interprets a **signed amount**: *positive* means
"more complex / more expressive" where that concept applies, *negative* means
"simpler / more regularized"; for parameters with no complexity meaning it is
plain numeric up/down (defined explicitly per parameter). Examples: `max_depth`/
`num_leaves` increase on positive; `min_child_samples` decreases on positive;
`lambda_l2`/`dropout` *reduce* regularization on positive; `learning_rate` is
defined as plain numeric movement.

```
numeric:    new = old · (1 + signed_amount)
log-scale:  new = old · exp(signed_amount)
integer:    mutate continuously, clamp to bounds, then stochastic-round
            (e.g. 7.3 → 7 w.p. 0.7, → 8 w.p. 0.3)
categorical: pick from allowed alternatives / neighbours
```

Validity is always enforced. For feature-reference genes, positive mutation adds
/ upgrades / increases feature count; negative removes / downgrades / decreases.
Removing a parent gene removes or adjusts its children.

### Model mutation (parent → child)

A `ModelMutator` selects a parent, picks a mutation *type*, mutates a **small
number of genes**, and emits a **new child `ModelGenome`** with a
`MutationRecord`; it validates the child, hashes it, and skips retraining if an
identical genome hash already exists. **The parent is never mutated in place.**

Rather than a flat per-gene 5% probability (which mutates too many genes in large
genomes and wrecks credit assignment), the number of mutated genes is drawn from
a distribution — `{1: 0.70, 2: 0.20, 3: 0.07, 4+: 0.03}` — and the amount from
`Uniform(drift − scale, drift + scale)` (`scale = 0.20`, `drift = 0.00`
initially; `drift` may later adapt to under/over-fitting and compute pressure).

Mutation types: `local_hyperparameter`, `coordinated_hyperparameter`,
`add_feature`, `remove_feature`, `replace_feature`, `change_feature_encoding`,
`model_family_restart` (a new genome, not an ordinary mutation), and planned
post-processing / ensemble-role mutations. Coordinated mutations are explicit and
named (e.g. higher `num_leaves` + higher `min_child_samples`; higher `max_depth`
+ stronger regularization).

### Resource genes and promotion

Resource/fidelity changes — more folds, more seeds, more trees/iterations, more
epochs, less row-subsampling — are treated as **promotion**, not ordinary
mutation. `ResourceGene`s are not mutated by behaviour mutation; a promising child
is *promoted* to a higher fidelity level (Fidelity 1: cheap holdout/1 fold →
Fidelity 4: multiple seeds/bagging). A model is **never** rewarded merely for
spending more compute — utilities are only compared within the same fidelity.

### Model scoring and utility

Scores live in an extensible `ModelScoreSet`. Initial: `score`, `score_std`,
`compute_time`; reserved: train/validation score, score gap, memory, inference
time, feature count, materialized width, model size, prediction diversity, fold
stability, public LB score, ensemble contribution, failure penalty. Every metric
declares whether higher is better and is converted to an internal
**larger-is-better** convention (`internal = −raw` for lower-is-better metrics).

```
adj_score      = score − 1.75 · score_std
t_ref          = median compute time over comparable trials
model_utility  = (adj_score − comparable_adj_score_avg)
                 / (1 + log(1 + compute_time / t_ref))
```

"Comparable" means the same competition, metric, validation scheme and **fidelity
level** — low-fidelity trials are never compared against full-fidelity ones.
Absolute score is tracked separately: an expensive model can be a poor *breeder*
yet a strong *final candidate*. Three rankings are maintained:
`efficient_search_ranking`, `absolute_score_ranking`, `ensemble_candidate_ranking`.

### Gene credit assignment

After a child trains, it is compared to its parent:

```
delta_utility = child_utility − parent_utility
delta_score   = child_score   − parent_score
delta_time    = child_time     − parent_time
behavior_delta = 1 − corr(parent_oof, child_oof)   # fallback: |delta_score| / note
```

Each mutated gene updates positive- or negative-mutation stats by the sign of its
`signed_amount`. A simple first credit is `gene_credit = delta_utility`; a better
form is `delta_utility · max(ε, behavior_delta)`, plus `compute_credit =
−delta_time`. Counts, means and variances are stored so one lucky mutation cannot
dominate.

### Populations, archives, and the controller

An `EvolutionController` runs each batch: generate & score feature candidates,
insert good ones and evict weak generated ones past the cap; decide whether to
**generate a new model** or **mutate an existing one** (with exploration floors —
neither probability drops below 0.1); build the genome; train; score; compute
utility; assign gene credit and downstream feature credit; update feature and
mutation probabilities; archive/promote; and record everything. Parent selection
uses **tournament selection** (sample `k=5`, pick best by utility plus small
diversity/recency bonuses) — it does not always mutate the single best model. An
`active_population` is eligible for mutation; an `elite_archive` keeps the best
ever for finalization and is not casually deleted. Per-family balance (and, later,
islands per model family) prevents one family dominating.

### Reproducibility, hashing and records

Reproducibility rests on canonical hashes and immutable records. The
**feature hash** is the canonical recipe; the **model genome hash** covers base
model, selected feature IDs, encoding choices, parameter values, resource/fidelity
settings, and (where relevant) validation-scheme and target-transform IDs and a
config/version tag. Before training, an existing genome hash short-circuits
duplicate work. `FeatureGenome`, `ModelGenome`, `Gene`, `FeatureScoreSet`,
`ModelScoreSet`, `MutationRecord` and `ModelResult` all serialize to
JSON-compatible structures.

### Validation and leakage discipline

Any transform that learns from the target is marked `uses_target=True` and
`requires_oof=True` and is materialized fold-safe. Target-dependent features are
never computed globally before validation. (Target encoding is stubbed for now.)

### How to extend

Every major object is built to grow, via dependency injection of evaluators,
scorers, transformations and model factories:

- **New transformation** → subclass `FeatureTransformation`, register it; the
  generator picks it up automatically.
- **New feature score** → add a `Score` to the `FeatureScoreSet`; combination is
  weight-driven, so nothing else changes.
- **New model family** → add a `BaseModelGene`/factory entry reusing the existing
  v1 model registry; parameter spaces are declared per family.
- **New model score / metric** → add to `ModelScoreSet` with its
  higher-is-better flag.
- **New mutation type, encoding, validation scheme or ensemble method** → add a
  named operator; controllers dispatch by name.

Each class has one main reason to change; prefer small explicit classes over
clever abstractions.

### KagglePipeline orchestration

`KagglePipeline` (in
[`kaggle_pipeline/evolution/pipeline.py`](src/kaggle_pipeline/evolution/pipeline.py))
is the main orchestration class. It owns the ecosystem state, feature registry,
model population, runtime manager, thread pools, the batch loop, checkpointing
and optional ensemble finalization. The pipeline runs in **batches**: each batch
may generate and score new features, create new model genomes, mutate existing
models into **child** models, train models in parallel, update scores and
mutation credit, print the ecosystem state, and save a checkpoint.

```python
from kaggle_pipeline.evolution import KagglePipeline

pipeline = KagglePipeline(
    max_runtime_hours=12,
    verbosity=3,
    enable_ensembling=True,
    num_workers=4,
    state_dir="state/my_competition",
    seed=0,
)
pipeline.fit(train_df, target="y", test_df=test_df)   # prepares + runs the loop
pipeline.make_submission("submission.csv")            # ensemble -> CSV
```

The pipeline is designed for long Kaggle-style runs. By default it respects a
**12-hour runtime limit** (measured with `time.monotonic`, not wall-clock). It
stops launching new training work before the deadline, saves the current
ecosystem state, and, if ensembling is enabled, reserves time to build an
ensemble from the best available models. If there is not enough time or not
enough candidate models for an ensemble, the pipeline falls back to the best
single model. It stops itself — it does not rely on an external timeout.

`KagglePipeline` is an *orchestrator*: the algorithms live in the small
collaborators (`FeatureRegistry`, `FeatureGenerator`, `ModelFactory`,
`ModelMutator`, `ModelTrainer`, `CreditAssigner`, `EvolutionController`,
`EnsembleManager`, `EcosystemSerializer`). Its public surface is `fit`, `run`,
`run_batch`, `ensemble`, `predict`, `make_submission`, `save_state`,
`load_state`, `checkpoint`, `print_state`, `summarize_state`, `shutdown`.

**Threading.** The main thread owns and mutates the ecosystem state; worker
threads run *pure* tasks (model training reads the registry/feature cache but
never mutates shared state) and return immutable result objects that the main
thread applies. A batch produces all genomes and pre-materialises their features
on the main thread, then trains in parallel, then applies results and assigns
credit on the main thread — so there are no races. Models train with `n_jobs=1`
while several run in parallel, avoiding CPU oversubscription.

**Checkpointing.** State is saved after every batch, before and after ensembling,
and on graceful interruption. Each checkpoint is an atomically-written directory
under `state_dir/checkpoints/` holding a pickled `EcosystemState` (registry,
population, OOF store, RNG state) plus JSON sidecars (`manifest.json`,
`config.json`, `summary.json`); a `latest.json` pointer tracks the most recent and
old checkpoints are pruned to `keep_last_n_checkpoints`. Checkpoints are taken at
**batch boundaries**, so the saved state only ever reflects results already
applied — there are no half-applied partial results, and no live thread pools or
futures are serialized. `load_state` restores the registries and RNG and rebuilds
the collaborators (the model-family callables are reconstructed from settings, not
unpickled).

**Verbosity** (0–4) controls how much the pipeline logs and prints, via the
standard `logging` module (thread-safe; no raw `print`):

| Level | Name | Prints |
| --- | --- | --- |
| 0 | SILENT | nothing routine (only critical errors) |
| 1 | SUMMARY | one-line batch summary (best score, counts, elapsed/remaining) |
| 2 | NORMAL | batch start/end, feature/model counts, checkpoints, best-model changes |
| 3 | DETAILED | top features, family stats, mutation success rates, runtime reserve |
| 4 | DEBUG | mutation types, gene-level detail, internal counters |

`print_state(detail_level=None)` prints the current ecosystem state at the given
level (defaulting to the configured verbosity); `summarize_state()` returns the
same information as a structured dict for logging or saving.

As everywhere in this design, **the parent model is never changed in place** —
mutation always creates a child model.

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
├── evolution/      # evolutionary rework (feature/model genomes, mutation, controller) — see "Evolutionary architecture"
├── training/       # the step loop under a time budget
├── submission/     # writes the submission CSV
├── pipeline.py     # run() — the training entry point
├── analysis.py     # analyze() — the standalone EDA entry point
└── cli.py          # `kaggle-pipeline run|analyze --config ...`
notebooks/          # kaggle_runner.ipynb — the thin notebook to upload to Kaggle
tests/              # unit + end-to-end smoke tests
```

## Quickstart on Kaggle

Upload [`notebooks/kaggle_runner.ipynb`](notebooks/kaggle_runner.ipynb), or paste
these cells. Enable **Internet** in the notebook settings.

```python
# 1. Install (pin a tag for reproducibility, e.g. @v0.1.0)
!pip install -q git+https://github.com/HkHk2Prod/Kaggle-Pipeline.git

# 2. Configure — on Kaggle most fields autodetect from the data, so a bare
#    Config() often works. Set only what you need to override per competition.
from kaggle_pipeline import Config
cfg = Config(
    # target / task / scoring / prediction_aim autodetect when left unset.
    feature_expressions=["new_flag = some_column < 25"],  # optional engineered columns
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
.venv/bin/kaggle-pipeline run --config path/to/config.yaml

# Or explore the data first (separate, no training). EDA is opt-in:
# set `run_eda: true` in the config -- otherwise analyze logs that it's
# disabled and exits without rendering.
.venv/bin/kaggle-pipeline analyze --config path/to/config.yaml

# Add -v for verbose output (per-model scores, full leaderboard) or -q to quiet it
.venv/bin/kaggle-pipeline run --config path/to/config.yaml -v
```

Or from Python:

```python
from kaggle_pipeline import Config, run, analyze

cfg = Config.from_yaml("path/to/config.yaml")
cfg.run_eda = True  # EDA is opt-in (run_eda defaults to False)
analyze(cfg)        # optional: render EDA only
run(cfg)            # train + write submission (ignores run_eda)
```

## Configuration

All knobs live in one `Config` object — set them in Python or load them from a
YAML file with `Config.from_yaml(path)` (keys map one-to-one onto the fields).
The most important fields:

| Field | Meaning |
| --- | --- |
| `competition` | Used to derive default Kaggle/Colab data paths |
| `target`, `id_col` | Target and id column name(s) |
| `task` | `classification` (the only implemented path). `regression` raises `NotImplementedError` — see [Regression](#regression). |
| `scoring` | `balanced_accuracy`, `roc_auc` or `neg_root_mean_squared_error` |
| `prediction_aim` | `category` or `probability` |
| `feature_expressions` | `pandas.eval` expressions for new features |
| `categorical_encoding` | Per-column encoding for non-native models (see below) |
| `prune_features`, `prune_alpha`, `redundancy_floor` | Auto-drop irrelevant / redundant predictors (see below) |
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

### Feature pruning

On by default (`prune_features`), the pretrain pipeline drops predictors that
carry no usable signal, with **size-inferred** thresholds:

- **Uncorrelated with the target** — a predictor whose association with the
  target is below `τ(n)`, the smallest correlation distinguishable from noise at
  level `prune_alpha`. `τ(n)` shrinks as the dataset grows, so big datasets keep
  weaker-but-real signals.
- **Redundant** — two predictors are collapsed (keeping the more
  target-relevant one) only when we are `1 − prune_alpha` confident their *true*
  association exceeds `redundancy_floor` (default 0.90), i.e. the lower end of a
  Fisher-z confidence interval clears the floor. Confidence (not just the point
  estimate) is what makes this size-aware.

Mixed types are handled via the same association measures as the EDA heatmap
(`|Pearson r|`, correlation ratio, Cramér's V).

**Data-quality alarm:** if a predictor looks uncorrelated with the target *yet*
is strongly correlated with another predictor that **is** target-relevant, that
breaks correlation transitivity (suppression, non-linearity, or leakage). The
predictor is **kept** and a loud `[prune] SUSPICIOUS …` warning is logged rather
than dropped. All drops are logged as `[prune] …` lines.

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

## Possible improvements

A running checklist of enhancements worth exploring — contributions welcome:

- [ ] **Check correlations during `add`, not just per batch.** De-correlation
  currently runs as a whole-board pass after each batch
  ([`search/decorrelation.py`](src/kaggle_pipeline/search/decorrelation.py)),
  which re-compares every pair even though the board is already de-correlated.
  Folding the residual-correlation check into `LeaderBoard.add` — rejecting a
  newcomer that duplicates a better kept model *before* it is admitted — would be
  cheaper (only newcomers vs. the kept set) and would stop a redundant model from
  evicting a diverse one to make room.
- [ ] **Nested cross-validation for less optimistic evaluation.** The same CV
  folds drive hyperparameter sampling, leaderboard scoring and the ensemble
  meta-model, so reported scores are mildly optimistic. An outer CV loop around
  the search would give a cleaner estimate of generalization.
- [ ] **A feature-engineering leaderboard.** Mirror the model leaderboard for
  *features*: track which engineered features (`feature_expressions`, encodings,
  interactions) actually help, and feed the winners back in so later search
  rounds build more diverse, complementary models instead of re-discovering the
  same signal.
- [ ] **Search on a subsample, refit winners on full data.** Run the model search
  on a smaller sample of the training set to evaluate many more candidates per
  unit time, then refit only the leaderboard survivors on the full dataset before
  ensembling.

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
- A single `Config.seed` controls all randomness: the model search, the
  leaderboard's class selection, the cross-validation fold shuffling and the
  ensemble search all derive from it. It defaults to `None` (non-reproducible,
  the original notebook's behaviour); set it to an int for a fully reproducible
  run (same seed -> same folds, leaderboard and submission).
- Progress output goes through Python's `logging` (the `kaggle_pipeline` logger)
  rather than `print`. How much is shown is the `verbosity` config field:
  `quiet` (warnings/errors only), `normal` (stage progress plus the autodetected
  fields, prune summary, each tested model's score and timing, chosen-ensemble
  score and submission path) or `verbose` (the default: adds the sampled
  per-model parameters, the full leaderboard after each step, the encoding plan
  and a submission preview). The CLI flags `-v`/`--verbose` and
  `-q`/`--quiet` override the config value, e.g.
  `kaggle-pipeline run -c config.yaml -v`. Embedders can still configure the
  `kaggle_pipeline` logger directly to redirect or filter output.

## License

MIT — see [LICENSE](LICENSE).
