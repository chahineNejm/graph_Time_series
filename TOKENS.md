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
