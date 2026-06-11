# Token Reference

This file is the living index of tokens in the project. Keep it practical:
when a new token is added, record what it reads, what it writes, whether it is
package-level or notebook-only, and how it changes `State`.

## Reading This File

Status meanings:

- `package`: importable from `graph_Time_series` or `graph_Time_series.token_blocks`.
- `notebook`: currently defined inside `examples/test_token_sequence_colab.ipynb`.
- `planned`: useful direction, not implemented yet.

State stores:

- `features`: compatibility/general store, currently includes `raw_history`.
- `historical_features`: sample-aligned model features.
- `future_features`: horizon-side known features.
- `metadata`: scalar/debug/config values.
- `flags`: lightweight booleans or guards.
- `artifacts`: inspectable metadata for named state values.
- `input_bundles`: inspectable metadata for bound model inputs.
- `prediction_stack`: model outputs in the active target space.

The core residual rule is:

```text
current_target = active_target_base - sum(prediction_stack)
```

Any model token that calls `state.push_prediction(...)` automatically updates
the residual for the next model.

## Package Tokens

These are the tokens currently promoted to the importable package.

### `ZNormalization`

Status: `package`

Class: `TransformToken`

File: `graph_Time_series/token_blocks/normalization.py`

Purpose:

Per-series z-normalization using each sample's history mean and standard
deviation. It also transforms the target into the same normalized space and
registers the inverse transform.

Reads:

- `raw_history`

Writes:

- `historical_features["scaled_history"]`
- `active_target_base`
- `transform_stack`
- `flags["is_z_normalized"]`

Hyperparameters in the current setting:

- no constructor hyperparameters
- implicit numerical guard: `sigma + 1e-8`
- state-dependent quantities: per-series history mean `mu` and standard
  deviation `sigma`

State effect:

```text
history -> scaled_history
future target -> normalized active_target_base
current_target -> normalized residual target
```

Common next tokens:

- `BindScaledHistory`
- `BindAllSafeTabular`
- `kernel_rbf`

### `MeanAbsScaling`

Status: `package`

Class: `TransformToken`

File: `graph_Time_series/token_blocks/normalization.py`

Purpose:

Scale each series by the mean absolute value of its history, without centering.
This preserves the zero point while putting series on a comparable magnitude
scale.

Reads:

- `raw_history`

Writes:

- `historical_features["scaled_history"]`
- `active_target_base`
- `transform_stack`
- `flags["is_mean_abs_scaled"]`

Hyperparameters in the current setting:

- no constructor hyperparameters
- implicit numerical guard: `mean(abs(history)) + 1e-8`
- state-dependent quantity: per-series mean absolute history scale

State effect:

```text
scale = mean(abs(history))
history -> history / scale
future target -> future target / scale
current_target -> scaled residual target
```

Common next tokens:

- `BindScaledHistory`
- `BindAllSafeTabular`
- `kernel_rbf`

### `BindScaledHistory`

Status: `package`

Class: `FeatureToken` with `token_class = "binding"`

File: `graph_Time_series/token_blocks/bindings.py`

Purpose:

Select `scaled_history` as the active model input. This is the simple, explicit
way to say: "the next model should consume the normalized sequence."

Reads:

- `scaled_history`

Writes:

- `historical_features["model_input"]`
- `features["model_input"]`
- `input_bundles["scaled_history"]`
- `metadata["active_input_bundle"]`

Bundle:

```text
name = "scaled_history"
kind = "sequence_flat"
artifact_names = ("scaled_history",)
```

Hyperparameters in the current setting:

- no constructor hyperparameters
- fixed artifact: `scaled_history`
- fixed bundle kind: `sequence_flat`

Common next tokens:

- `kernel_rbf`

### `BindAllSafeTabular`

Status: `package`

Class: `FeatureToken` with `token_class = "binding"`

File: `graph_Time_series/token_blocks/bindings.py`

Purpose:

Flatten all finite historical artifacts into one tabular bundle. This is useful
for quick tests, but it can make the search space and feature width large.

Reads:

- all finite entries in `historical_features`, except `model_input`

Writes:

- `historical_features["model_input"]`
- `features["model_input"]`
- `input_bundles["all_safe_tabular"]`
- `metadata["active_input_bundle"]`

Bundle:

```text
name = "all_safe_tabular"
kind = "tabular"
artifact_names = all selected historical feature names
```

Hyperparameters in the current setting:

- no constructor hyperparameters
- state-dependent feature set: every finite entry in `historical_features`
  except `model_input`
- fixed bundle kind: `tabular`

Use carefully:

This token is broad by design. Prefer a targeted `StackFeatureBundleToken` when
you already know which artifacts should be combined.

### `BindFeatureToken`

Status: `package`

Class: `FeatureToken` with `token_class = "binding"`

File: `graph_Time_series/token_blocks/bindings.py`

Purpose:

Generic single-artifact binder. It lets us create named binders such as
`BindRawHistory`, `BindLevelSeries`, or `BindShapeVector` without writing a new
class each time.

Configured example:

```python
BindFeatureToken(
    "raw_history",
    token_name="BindRawHistory",
    bundle_name="raw_history",
    kind="sequence_flat",
)
```

Reads:

- one configured artifact name

Writes:

- `model_input`
- active input bundle metadata

Hyperparameters in the current setting:

- `feature_name`: artifact to read
- `token_name`: visible token name, default `Bind[{feature_name}]`
- `bundle_name`: active bundle name, default `feature_name`
- `kind`: input bundle kind, default `sequence_flat`
- `target_space`: default `active_target`

Notebook-configured examples:

- `BindRawHistory`: `feature_name="raw_history"`, `kind="sequence_flat"`
- `BindFourierFeatures`: `feature_name="fourier_features"`, `kind="tabular"`

### `StackFeatureBundleToken`

Status: `package`

Class: `FeatureToken` with `token_class = "binding"`

File: `graph_Time_series/token_blocks/bindings.py`

Purpose:

Flatten and concatenate a chosen set of artifacts into one tabular model input.
This is the controlled version of "multi-output from previous tokens becomes
multi-input for the next model."

Configured example:

```python
StackFeatureBundleToken(
    ("scaled_history", "context_history"),
    token_name="StackScaledContext",
    bundle_name="scaled_plus_context",
)
```

Reads:

- configured artifact names

Writes:

- `model_input`
- active input bundle metadata

Bundle:

```text
kind = "tabular"
metadata["component_shapes"] = original shapes before flattening
```

Hyperparameters in the current setting:

- `feature_names`: tuple of artifacts to flatten and concatenate
- `token_name`: visible token name, default `Stack[a+b+...]`
- `bundle_name`: active bundle name, default joined feature names
- `kind`: default `tabular`
- `target_space`: default `active_target`

Notebook-configured examples:

- `StackScaledContext`: `("scaled_history", "context_history")`
- `StackFourierScaled`: `("fourier_features", "scaled_history")`

### `PeriodPhaseOneHot`

Status: `package`

Class: `FeatureToken`

File: `graph_Time_series/token_blocks/periodic.py`

Purpose:

Encode positions inside a selected period. This token expects a previous period
selection step to have written `metadata["period"]`.

Reads:

- `raw_history`
- `metadata["period"]`

Writes:

- `historical_features["history_period_phase_id"]`
- optionally `historical_features["history_period_phase_onehot"]`
- `future_features["future_period_phase_id"]`
- `future_features["future_period_phase_onehot"]`
- `metadata["period_phase_onehot"]`
- `flags["period_phase_encoded"]`

Hyperparameters in the current setting:

- `anchor="fold"`: align phases with `PeriodFold`
- `include_history_onehot="auto"`: write history one-hot only under the memory
  guard
- `max_history_onehot_cells=20_000_000`
- `max_future_onehot_cells=50_000_000`
- `dtype=np.float32`
- state-dependent quantity: `metadata["period"]`

State effect:

```text
period = metadata["period"]
history_period_phase_id:      (n_samples, history_length)
history_period_phase_onehot:  (n_samples, history_length, period), when small enough
future_period_phase_id:       (n_samples, horizon)
future_period_phase_onehot:   (n_samples, horizon, period)
```

Default alignment:

`anchor="fold"` matches `PeriodFold`: trailing complete periods define phase
zero, so the first forecast step starts at phase zero. Use
`anchor="history_start"` if you want phases measured from the start of the
current history array.

Memory guard:

The token always writes compact phase IDs. History-side one-hot is automatic
only when it is below `max_history_onehot_cells`; this avoids exploding memory
on long histories. Future-side one-hot is required, but `can_apply()` refuses if
it would exceed `max_future_onehot_cells`.

Typical use:

```text
PeriodSelection -> PeriodPhaseOneHot
```

Current limitation:

The existing model tokens mostly consume `historical_features["model_input"]`.
The future one-hot is stored correctly as a known future feature, but a future
covariate-aware model/binder is still a planned next step.

### `FourierFeatures`

Status: `package`

Class: `FeatureToken`

File: `graph_Time_series/token_blocks/fourier.py`

Purpose:

Create fixed-width tabular FFT features from the history. This is meant for
tabular models such as random forests and LightGBM.

Reads:

- preferred source: `historical_features["scaled_history"]`
- fallback: `features["raw_history"]`

Writes:

- `historical_features["fourier_features"]`
- `metadata["fourier_features"]`

Hyperparameters in the current setting:

- `n_harmonics=16`: number of low-frequency FFT bins requested
- `source_feature="scaled_history"`: preferred history artifact
- `fallback_to_raw=True`: use `raw_history` if `scaled_history` is absent
- `include_summary=True`: append mean, std, min, max, last, and slope
- `center=True`: subtract the per-series history mean before FFT
- state-dependent quantity: actual bins are capped by available history length

State effect:

```text
history -> low-frequency FFT real/imag/amplitude/relative-power features
history -> summary features such as mean, std, min, max, last, slope
fourier_features: (n_samples, 4 * n_harmonics + summary_width)
```

Typical use:

```text
MeanAbsScaling -> FourierFeatures -> BindFourierFeatures -> rf_tabular
MeanAbsScaling -> FourierFeatures -> BindFourierFeatures -> lightgbm_tabular
```

Notes:

The token uses fixed low-frequency bins rather than data-selected top bins, so
train and holdout states keep the same feature width.

### `kernel_rbf`

Status: `package`

Class: `ModelToken`

File: `graph_Time_series/token_blocks/kernel_rbf.py`

Purpose:

RBF KernelRidge model using the active model input. If no explicit model input
has been bound, it falls back to `scaled_history`.

Reads:

- `historical_features["model_input"]`, or fallback `scaled_history`
- `current_target`

Writes:

- `prediction_stack`
- `prediction_names`
- `features["last_prediction"]`
- `features["cumulative_prediction"]`
- `features["current_residual"]`
- `metadata["last_prediction"]`
- `metadata["cumulative_prediction"]`
- `metadata["current_residual"]`

Accepted bundle kinds:

- `sequence_flat`
- `tabular`

Hyperparameters in the current setting:

- `alpha=1e-2`: KernelRidge regularization
- `median_subset=24`: maximum number of samples used for the median-distance
  lengthscale heuristic
- `seed=0`: subset sampling seed
- derived parameter: `gamma = 1 / (2 * lengthscale ** 2)`

State effect:

```text
fit model on current_target
push prediction
current_target becomes residual
```

Note:

The package token performs leave-one-out predictions inside the current state.
For true holdout comparison, use the notebook's leakage-safe wrappers or a
future package evaluator that separates train and query states.

### `rf_tabular`

Status: `package`

Class: `ModelToken`

File: `graph_Time_series/token_blocks/tabular_models.py`

Purpose:

Random forest regressor over the active tabular `model_input`. It supports
multi-horizon targets directly through scikit-learn's multi-output regression.

Reads:

- `historical_features["model_input"]`
- active input bundle with `kind = "tabular"`
- `current_target`

Writes:

- `prediction_stack`
- residual updates through `state.push_prediction(...)`

Hyperparameters in the current setting:

- `n_estimators=120`
- `max_depth=12`
- `min_samples_leaf=2`
- `seed=0`
- `n_jobs=-1`

Use carefully:

Direct token application fits and predicts on the same state for diagnostics.
Notebook comparison uses leakage-safe train/query wrappers.

### `lightgbm_tabular`

Status: `package`

Class: `ModelToken`

File: `graph_Time_series/token_blocks/tabular_models.py`

Purpose:

LightGBM regressor over active tabular `model_input`. It is the gradient
boosted tree tabular baseline for Fourier and other tabular feature bundles.
The package import is lazy: installing the project does not require LightGBM
until this token is actually built or run.

Reads:

- `historical_features["model_input"]`
- active input bundle with `kind = "tabular"`
- `current_target`

Writes:

- `prediction_stack`
- residual updates through `state.push_prediction(...)`

Hyperparameters in the current setting:

- `n_estimators=200`
- `learning_rate=0.05`
- `num_leaves=31`
- `max_depth=-1`
- `min_child_samples=20`
- `subsample=0.9`
- `colsample_bytree=0.9`
- `seed=0`
- `n_jobs=-1`
- wrapper: `MultiOutputRegressor(LGBMRegressor(...))` for multi-horizon
  targets

Dependency:

- `lightgbm`, installed by the Colab setup cell for notebook comparisons

Use carefully:

Direct token application fits and predicts on the same state for diagnostics.
Notebook comparison uses leakage-safe train/query wrappers.

## Notebook Prototype Tokens

These tokens currently live in `examples/test_token_sequence_colab.ipynb`.
They are useful, but not yet promoted to package files.

### `DataAugmentation`

Status: `notebook`

Class: `TransformToken`

Purpose:

Apply simple history augmentation such as jitter and smoothing. In leakage-safe
evaluation, it is treated as training-only and skipped for query/holdout states.

Reads:

- `raw_history`

Writes:

- augmented history values
- augmentation metadata

Hyperparameters in the current setting:

- `jitter_sigma=0.03`: noise scale as a fraction of each series std
- `smooth_window=5`: moving-average window; values `<= 1` disable smoothing
- `seed=42`: random seed for jitter

Use carefully:

This token must not learn from holdout futures. It should only modify histories
in a train-safe way.

### `ContextWindow`

Status: `notebook`

Class: `CleaningToken`

Purpose:

Keep only the latest `context_length` history points.

Reads:

- `raw_history`

Writes:

- `features["raw_history"]`
- `historical_features["context_history"]`
- `metadata["context_length"]`

Hyperparameters in the current setting:

- `context_length=min(720, H.shape[1])` in the notebook registry
- constructor default: `context_length=720`

State effect:

```text
original_history shape: (n_samples, full_history)
after token:            (n_samples, context_length)
```

Example from the current notebook:

```text
context_length = 720
```

### `PeriodSelection`

Status: `notebook`

Class: `FeatureToken`

Purpose:

This is the token that recognizes/selects a period. It does not forecast by
itself. It looks at the history, scores candidate periods, and stores the
selected period in state metadata.

Reads:

- `raw_history`

Writes:

- `metadata["period"]`
- `metadata["period_scores"]`
- `flags["period_selected"]`

Hyperparameters in the current setting:

- `freq="H"` in the notebook registry
- `max_series=8`: number of series used to score candidate periods
- state-dependent filter: candidates need at least three complete cycles in
  the current history window

How it works right now:

1. It chooses candidate periods from a frequency map.
2. For hourly data, candidates include values such as daily and weekly periods.
3. It keeps candidates with enough complete cycles in the available history.
4. It scores them with a simple BIC/MDL-style reconstruction criterion.
5. It stores the lowest-scoring period as `metadata["period"]`.

Current frequency map:

```text
H   -> [24, 168]
D   -> [7, 365]
W   -> [52]
M   -> [12]
15T -> [4, 96]
5T  -> [12, 288]
```

Typical next token:

- `PeriodPhaseOneHot`
- `PeriodFold`

Important distinction:

`PeriodSelection` identifies a period. `PeriodFold` uses that period to reshape
the history into phase/cycle structure.

### `PeriodFold`

Status: `notebook`

Class: `FeatureToken`

Purpose:

Fold the history into period phases and complete cycles. This creates the shape
needed by shape/level tokens.

Reads:

- `raw_history`
- `metadata["period"]`

Writes:

- `historical_features["period_matrix"]`
- `historical_features["level_series"]`
- `metadata["n_complete_periods"]`

Hyperparameters in the current setting:

- no constructor hyperparameters
- state-dependent quantity: `metadata["period"]`
- state-dependent quantity: number of complete periods available in history

State effect:

```text
raw_history:    (n_samples, history_length)
period_matrix:  (n_samples, period, n_complete_periods)
level_series:   (n_samples, n_complete_periods)
```

Typical next token:

- `ShapeLevel`

### `ShapeLevel`

Status: `notebook`

Class: `FeatureToken`

Purpose:

Estimate a within-period shape vector using recent folded periods. This creates
a normalized phase profile that can distribute level forecasts across the
horizon.

Reads:

- `period_matrix`
- `level_series`

Writes:

- `historical_features["shape_vector"]`
- `metadata["shape_k"]`

Hyperparameters in the current setting:

- `shape_k=2`: number of most recent complete periods used to estimate shape
- state-dependent cap: `shape_k` is clipped to available complete periods

State effect:

```text
shape_vector: (n_samples, period)
```

Typical next tokens:

- `shape_naive`
- `level_kernel_rbf`

### `shape_naive`

Status: `notebook`

Class: `ModelToken`

Purpose:

Forecast by taking the last observed level and spreading it over future phases
using the frozen shape vector.

Reads:

- `shape_vector`
- `level_series`

Writes:

- `prediction_stack`
- residual updates through `state.push_prediction(...)`

Hyperparameters in the current setting:

- no constructor hyperparameters
- state-dependent quantities: last value of `level_series`, `shape_vector`,
  and forecast horizon

Pipeline:

```text
PeriodSelection -> PeriodFold -> ShapeLevel -> shape_naive
```

### `kernel_rbf_fast`

Status: `notebook`

Class: `ModelToken`

Purpose:

Fast KernelRidge diagnostic token. It fits once and predicts on the same state,
so it is useful for manual inspection but not safe as a direct holdout metric.

Reads:

- active `model_input`, or fallback `scaled_history`
- `current_target`

Writes:

- `prediction_stack`
- residual updates
- kernel metadata such as lengthscale and gamma

Hyperparameters in the current setting:

- `alpha=1e-2`
- `median_subset=24`
- `seed=0`
- derived parameter: `gamma = 1 / (2 * lengthscale ** 2)`

Use carefully:

Manual in-state metrics from this token can look very good because they are
in-sample. For comparison, use `safe_score_sequence(...)` from the notebook.

### `level_kernel_rbf`

Status: `notebook`

Class: `ModelToken`

Purpose:

Fit an RBF KernelRidge model on `level_series`, then expand predicted future
levels through `shape_vector`.

Reads:

- `level_series`
- `shape_vector`
- `current_target`

Writes:

- `prediction_stack`
- residual updates
- kernel metadata

Hyperparameters in the current setting:

- `alpha=1e-2`
- `median_subset=24`
- `seed=0`
- derived parameter: `gamma = 1 / (2 * lengthscale ** 2)`
- state-dependent quantity: number of future level blocks
  `ceil(horizon / period)`

Pipeline:

```text
PeriodSelection -> PeriodFold -> ShapeLevel -> level_kernel_rbf
```

Known issue:

The direct notebook token is in-state. The leakage-safe evaluator has a separate
train/query wrapper, but some current candidate combinations can fail when
train and query feature widths differ after period folding.

## Common Token Sequences

### Simple Normalized Kernel

```text
ZNormalization -> BindScaledHistory -> kernel_rbf
```

Meaning:

Normalize the history and target, bind the normalized sequence, fit an RBF model.

### Current Notebook Manual Sequence

```text
ContextWindow -> ZNormalization -> BindScaledHistory -> kernel_rbf_fast
```

Meaning:

Use the last context window, normalize it, bind the normalized sequence, run the
fast RBF diagnostic model.

### Shape Naive Period Pipeline

```text
PeriodSelection -> PeriodFold -> ShapeLevel -> shape_naive
```

Meaning:

Select a period, fold history into phases/cycles, estimate a shape, repeat the
last level through that shape.

### Period Phase Encoding

```text
PeriodSelection -> PeriodPhaseOneHot
```

Meaning:

Select a period, then add compact phase IDs plus one-hot encodings of positions
inside that period. This is useful for future covariate-aware models and for
controlled tabular experiments after a context window.

### Fourier Tabular Models

```text
MeanAbsScaling -> FourierFeatures -> BindFourierFeatures -> rf_tabular
MeanAbsScaling -> FourierFeatures -> BindFourierFeatures -> lightgbm_tabular
```

Meaning:

Scale the history, extract fixed-width Fourier features, bind them as tabular
input, then fit a tabular regressor on the transformed target.

Notebook comparison knobs:

- `TABULAR_MAX_SAMPLES = min(160, H.shape[0])`
- `TABULAR_MAX_HOLDOUT = min(40, H_holdout.shape[0])`
- `RUN_TABULAR_CV = True`
- `TABULAR_CV_FOLDS = 3`
- `TABULAR_CV_SEED = 11`

## STL Decomposition Proposal

Status: `planned`

No STL token is implemented yet. A good design would keep STL as a family of
small composable tokens rather than one giant block.

Potential token split:

- `STLDecomposition`: reads `raw_history` and `metadata["period"]`, writes
  `stl_trend`, `stl_seasonal`, and `stl_residual` historical artifacts.
- `STLTrendFeatures`: turns trend into slope, last trend level, curvature, and
  recent trend change tabular features.
- `STLSeasonalFeatures`: summarizes the seasonal component; can combine with
  `PeriodPhaseOneHot` or `FourierFeatures`.
- `STLResidualFeatures`: computes residual volatility, autocorrelation, recent
  residual windows, and residual Fourier features.
- `STLSeasonalNaive`: model token that repeats the learned seasonal profile
  into the horizon.
- `STLResidualModel`: model token or sequence that fits the remaining residual
  with `rf_tabular`, `lightgbm_tabular`, or `kernel_rbf`.

How it would compose with existing tokens:

```text
PeriodSelection -> STLDecomposition -> STLSeasonalNaive
PeriodSelection -> STLDecomposition -> STLTrendFeatures -> BindAllSafeTabular -> rf_tabular
PeriodSelection -> STLDecomposition -> STLResidualFeatures -> FourierFeatures -> lightgbm_tabular
```

Important constraints:

- STL must be fit on history only.
- Period should come from `PeriodSelection` or a user-supplied period.
- Seasonal forecast and residual forecast should enter through
  `state.push_prediction(...)`, so residual stacking remains correct.
- For leakage-safe evaluation, STL decomposition must be applied separately to
  train and query histories.

### Level Kernel Period Pipeline

```text
PeriodSelection -> PeriodFold -> ShapeLevel -> level_kernel_rbf
```

Meaning:

Select a period, create level and shape features, predict future levels with an
RBF kernel, then expand levels back into the forecast horizon.

## How To Add A Token Entry

Use this template:

```text
### `TokenName`

Status: `package`, `notebook`, or `planned`

Class: `CleaningToken`, `FeatureToken`, `TransformToken`, `ModelToken`, or `ControlToken`

Purpose:

One or two sentences.

Reads:

- state key or artifact

Writes:

- state key or artifact

Hyperparameters in the current setting:

- parameter name and default
- state-dependent quantities, if any

State effect:

Short description of how the state changes.

Typical next tokens:

- token name

Notes:

Leakage risks, shape constraints, bundle kinds, or search implications.
```

## Planned Token Families

These are open-ended possibilities. Add concrete entries above only when they
become real package tokens or notebook prototypes.

| Family | Possible tokens |
| --- | --- |
| Context selection | adaptive window, multi-resolution context |
| Period features | robust period search, multiple periods, period confidence |
| Shape/level | residual shape correction, multi-shape mixtures |
| Tabular models | ridge, random forest, gradient boosting, XGBoost |
| Sequence models | multi-channel binders, convolutional models, patch models |
| Future covariates | calendar binders, known-price features, holidays |
| Residual ensembles | model chains that deliberately fit remaining residuals |
| Kernel variants | DTW, shape-only, level-only, hybrid shape-level kernels |
| Search | grammar enumeration, MCTS, cost-aware candidate pruning |
| AST/config | keyword-configured tokens, config-file token registry |
