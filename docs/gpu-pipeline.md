# GPU pipeline (planned)

> Status: **planned / not implemented.** This is a design sketch for an opt-in
> pipeline that runs on Kaggle's GPU accelerators. Tracked from the *Possible
> improvements* checklist in the [README](../README.md#possible-improvements).

## Goal

An opt-in, parallel training cycle that runs on Kaggle's GPU accelerators rather
than only CPU. The current MLP family runs single-threaded on CPU, which is why
it carries a `max_train_rows` cap; the natural first cut is, after the CPU search
has filled the leaderboard with tree/linear/Bayesian models, to kick off a
separate GPU-only cycle that trains *only* the families that actually benefit
from a GPU. Search ranks and ensemble slots still decide whether those models
make it in, so GPU compute is reserved for the model types that benefit instead
of sitting idle while CPU-friendly GBMs train, and NN quality is not artificially
capped by CPU budget.

**What benefits from a GPU.** A PyTorch NN family (alongside or replacing
sklearn's MLP) is the clear win. GPU builds of the GBMs (`device="cuda"` /
`tree_method="gpu_hist"` for XGBoost/LightGBM/CatBoost) are *secondary/optional*:
at the tabular sizes here, on a Kaggle T4 they are often no faster than CPU
`hist` and add determinism/portability cost, so treat them as a later add-on, not
the headline.

**Out of scope:** TPUs. They use a different (XLA/JAX) programming model; this
design targets CUDA GPUs only.

## Configuration surface

Detection is automatic (see below); the config only needs an **override knob**
rather than a separate enable flag. One `device` setting both forces and caps:

- `device: auto` (default) — autodetect and use a GPU if present, else CPU.
- `device: cpu` — force the CPU pipeline even on a GPU host.
- optional manual caps (VRAM, compute capability) layered on top.

The override doubles as a **testability hook**: it lets the autodetect →
parameter logic be exercised on a CPU-only CI machine by feeding it a spoofed
capability ("pretend this is a 16 GB T4"), instead of the logic only being
runnable on real GPU hardware.

## Hardware autodetection

Detect GPU capability at startup and adapt model parameters to it — no user
input required:

- PyTorch: `torch.cuda.is_available()`, `torch.cuda.get_device_properties(0)`
  (name, total VRAM, compute capability, SM count), `torch.cuda.device_count()`.
- or parse `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv`.

From that, derive knobs like batch size, model width/depth caps, and whether to
keep folds on device, so the same code scales from a Kaggle T4 (16 GB) to a P100
without hand-tuning. Gracefully fall back to the CPU pipeline when no GPU is
present, so configs stay portable.

**Open question — multi-GPU (2×T4).** Decide the policy: one model per GPU
(parallelism across the search) vs. one model split across both via
DataParallel/DDP. The detection layer should expose device count so either is
possible.

## Kaggle time budget

Kaggle caps GPU notebook sessions differently from CPU ones — a separate weekly
GPU quota *and* a per-session wall-clock limit. The time-budget accounting that
bounds the search must read the **accelerator** limit when running on GPU instead
of assuming the CPU notebook ceiling; otherwise a GPU run either stops early or
overruns its session.

## Cross-host artifact contract

This is the core correctness constraint. A model trained on a GPU (or on a
*stronger* GPU than the current host) may not be recomputable here, so the
pipeline must treat its outputs as frozen artifacts rather than something it can
regenerate on demand.

- **Persist predictions.** Store each GPU model's OOF/test predictions as
  artifacts (not just the genome/recipe). Otherwise a later CPU merge/submit
  notebook can't reconstruct them and the model vanishes from the ensemble.
- **Tag models with `system_requirements`.** A declared `system_requirements`
  (needs-GPU, min VRAM, min compute capability, device count, …) gates
  *(re)training and evaluation*. If the current host can't satisfy them, the
  model is **not trained or evaluated** here — e.g. a model trained on a stronger
  GPU must not be re-evaluated on a weaker one, since it would OOM or silently
  change behaviour.
- **Reuse artifacts freely.** The requirement check gates recomputation, **not
  consumption**. If we already have what we need (e.g. stored predictions for the
  ensemble/submission), a weaker host may use those artifacts without satisfying
  the original training requirements. Only recomputation is restricted.
- **Don't chase GPU determinism (by design).** Don't try to make GPU results
  bit-reproducible to satisfy the hashing/records system. Determinism is only
  achievable on the *same* GPU + library/driver versions (and even then costs
  speed and bans some non-deterministic ops); *across* different GPUs it is
  impossible — different architectures pick different kernels and reduction
  orders, so results differ even with identical code and seeds. That is exactly
  why cross-host safety comes from `system_requirements` + stored artifacts, not
  deterministic flags. Record artifact **provenance** (GPU name, driver, library
  versions) so a genome-hash match isn't mistaken for prediction-identical.
- **The merge must tolerate a no-op GPU cycle.** If the GPU checkpoint
  contributes zero models (all rank below CPU ones, or the host couldn't run
  them), the submission must still be valid from the CPU leaderboard alone.

## Blendable sampling across notebooks

Today OOF predictions only blend across notebooks when they share
`search_sample_seed`, and the merge *recomputes* them when subsamples differ
([`_ensure_oof_compatible`](../src/kaggle_pipeline/evolution/pipeline.py)) — a
path a CPU merge host can't take for a GPU-only model.

- **Dedicated subsample seed.** Give the subsample its own seed so the sampled
  row set is reproducible independent of the evolution seed.
- **Nested (superset) sampling for NN models.** NN families often want *more*
  rows than the CPU GBM cap, so the GPU sample should be a **superset** of the
  CPU one (nested/hierarchical: the smaller sample's rows are a subset of the
  larger). OOF predictions then still line up on the shared rows and stay
  blendable.
- **Warn on size mismatch.** Log a warning whenever sample sizes differ, so a
  mismatch is visible rather than silently dropping the un-recomputable model
  from the blend.
- **Test-prediction alignment.** Beyond OOF, *test* predictions must align on a
  stable row-id/order across notebooks for the submission blend to be valid.

## Engineering hygiene

- **Install CUDA libraries only on GPU runs.** The heavy GPU dependencies
  (`torch`, GPU builds of XGBoost/LightGBM/CatBoost) live behind an install extra
  (e.g. `.[gpu]`) and lazy imports — installed and imported only when a GPU run
  is requested, never on CPU runs, so the CPU path stays light and imports
  cleanly without them.
- **Isolate runtime OOM per model.** Autodetected caps lower the risk but
  training can still hit CUDA out-of-memory unpredictably. Catch it per model —
  log a warning, mark that model failed (or retry once with a smaller batch) —
  instead of letting one OOM kill the whole notebook and lose the run.

## Needs design

- **Hardware-aware compute-cost accounting.** The utility / compute-waste
  accounting and resource genes weigh compute, but GPU-seconds are not
  CPU-seconds and draw from a *separate, scarce weekly GPU quota*. Without a
  hardware-aware cost notion, GPU models look cheap on wall-clock while burning
  the quota that actually limits throughput. Decide how to price GPU vs. CPU time
  in utility before the GPU cycle competes for budget against CPU models.
