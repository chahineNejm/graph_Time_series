# Token Reference

This is the living index of implemented tokens. It should stay aligned with the
package code and the current signal-board architecture.

Current rule: package models should work without binder/glue tokens. They use
`state.feature_bundle()` whenever no legacy `historical_features["model_input"]`
has been created manually.

Do not add or modify tokens without explicit approval.

## State Stores

`State` keeps both legacy stores and the new semantic board:

| Store | Purpose |
| --- | --- |
| `features` | Compatibility/general values, including `raw_history` |
| `historical_features` | Sample-aligned history-side artifacts |
| `future_features` | Horizon-side known features |
| `metadata` | Scalar/debug/config values |
| `flags` | Lightweight guards |
| `artifacts` | Inspectable metadata for named values |
| `board` | Typed `Signal` objects emitted by artifacts/transforms/models |
| `prediction_stack` | Model outputs in the active target space |

The residual rule is:

```text
current_target = active_target_base - sum(prediction_stack)
```

Any model token that calls `state.push_prediction(...)` updates the residual for
the next model.

## Signal Concepts

Common semantics:

| Sem | Meaning |
| --- | --- |
| `series` | History-like target values |
| `features` | Model covariates or flattened feature blocks |
| `level` | Per-period aggregate/scale |
| `shape` | Within-period profile |
| `forecast` | Point prediction |
| `samples` | Stochastic paths |
| `mask` | Validity or cleaning mask |
| `param:*` | Scalar or structured parameters |

Common axes:

```text
sample, time, horizon, feature, phase, period_idx, path
```

`state.feature_bundle()` adapts compatible sample-leading signals into one
`(sample, feature)` matrix and records the component blocks.

## Removed Legacy Tokens

The previous package binder tokens are no longer active package tokens:

```text
BindScaledHistory
BindAllSafeTabular
BindFeatureToken
StackFeatureBundleToken
```

`graph_Time_series/token_blocks/bindings.py` is a removal stub. Old notebooks
that create `model_input` manually may still work because models keep a
backward-compatible `model_input` path, but new package examples should not
require binders.

Future selector tokens may be useful, but they should be designed as optional
search moves that narrow/tag the auto-bundle, not as required plumbing.

## Registry Summary

### Default

Registered by `register_default_tokens(grammar)`:

```text
ZNormalization
MeanAbsScaling
FourierFeatures
kernel_rbf
rf_tabular
lightgbm_tabular
```

### FLAIR

Registered by `register_flair_tokens(grammar)`:

```text
FlairPreprocess
PeriodSelection
PeriodPhaseOneHot
PeriodFold
LevelShrinkage
ShapeLevel
SecondaryLevelSeasonality
LevelBoxCoxCenter
FlairRidgeLevel
FlairSamplePaths
```

### Versatile

Registered by `register_versatile_tokens(grammar)`:

```text
DayOfWeekFeature
versatile_rf
versatile_gb
```

### FLAIR GB Swap

Registered by `register_flair_gb_swap(grammar)`:

```text
gb_level_forecast
```

With all registries enabled:

```text
Grammar(20 tokens, 58 edges)
```

## Package Tokens

### `ZNormalization`

Status: `package`

Class: `TransformToken`

File: `graph_Time_series/token_blocks/normalization.py`

Purpose: per-series z-normalization using history mean and standard deviation.

Reads:

- `raw_history`

Writes:

- `historical_features["scaled_history"]`
- `active_target_base`
- transform stack
- `flags["is_z_normalized"]`

Hyperparameters:

- no constructor hyperparameters
- numerical guard: `sigma + 1e-8`
- state-dependent `mu` and `sigma`

State effect:

```text
history -> (history - mu) / sigma
future target -> (future - mu) / sigma
final prediction inverse -> pred * sigma + mu
```

Common next tokens:

- `FourierFeatures`
- `kernel_rbf`
- `rf_tabular`
- `lightgbm_tabular`
- `DayOfWeekFeature`
- `versatile_rf`
- `versatile_gb`

### `MeanAbsScaling`

Status: `package`

Class: `TransformToken`

File: `graph_Time_series/token_blocks/normalization.py`

Purpose: scale each series by mean absolute history value, without centering.

Reads:

- `raw_history`

Writes:

- `historical_features["scaled_history"]`
- `active_target_base`
- transform stack
- `flags["is_mean_abs_scaled"]`

Hyperparameters:

- no constructor hyperparameters
- numerical guard: `mean(abs(history)) + 1e-8`
- state-dependent per-series scale

State effect:

```text
history -> history / scale
future target -> future / scale
final prediction inverse -> pred * scale
```

Common next tokens:

- `FourierFeatures`
- `kernel_rbf`
- `rf_tabular`
- `lightgbm_tabular`
- `DayOfWeekFeature`
- `versatile_rf`
- `versatile_gb`

### `FourierFeatures`

Status: `package`

Class: `FeatureToken`

File: `graph_Time_series/token_blocks/fourier.py`

Purpose: create fixed-width FFT/tabular features from history.

Reads:

- preferred `scaled_history`
- fallback `raw_history`

Writes:

- `historical_features["fourier_features"]`
- artifact metadata with tags `fourier`, `tabular`
- `metadata["fourier_features"]`

Hyperparameters:

- `n_harmonics=16`
- `source_feature="scaled_history"`
- `fallback_to_raw=True`
- `include_summary=True`
- `center=True`

State effect:

```text
history sequence -> real/imag/amplitude/relative-power FFT blocks
optional summaries -> mean, std, min, max, last, slope
```

Common next tokens:

- `kernel_rbf`
- `rf_tabular`
- `lightgbm_tabular`
- `versatile_rf`
- `versatile_gb`
- `DayOfWeekFeature`

### `kernel_rbf`

Status: `package`

Class: `ModelToken`

File: `graph_Time_series/token_blocks/kernel_rbf.py`

Purpose: RBF KernelRidge model with median-heuristic lengthscale and
leave-one-out in-state predictions.

Reads:

- legacy `historical_features["model_input"]` if present;
- else `state.feature_bundle()`;
- fallback `scaled_history` for older direct usage;
- `current_target`.

Writes:

- `prediction_stack[-1]`
- execution log values for `lengthscale` and `gamma`

Hyperparameters:

- `alpha=1e-2`
- `median_subset=24`
- `seed=0`

State effect:

```text
X = model_input or feature_bundle or scaled_history
Y = current_target
prediction -> state.push_prediction(...)
current_target becomes residual
```

Common next tokens:

- `kernel_rbf`
- `STOP`

### `rf_tabular`

Status: `package`

Class: `ModelToken`

File: `graph_Time_series/token_blocks/tabular_models.py`

Purpose: RandomForestRegressor over the active tabular feature bundle.

Requires:

- `Port(sem="features", axes=("sample", "feature"), space="current", multiple=True, coerce=True)`

Reads:

- legacy `model_input` if present and tabular;
- else `state.feature_bundle()`;
- `current_target`.

Writes:

- `prediction_stack[-1]`
- `metadata["rf_tabular"]`

Hyperparameters:

- `n_estimators=120`
- `max_depth=12`
- `min_samples_leaf=2`
- `seed=0`
- `n_jobs=-1`

State effect:

```text
feature bundle -> RandomForestRegressor -> residual prediction
```

Common next tokens:

- `STOP`

### `lightgbm_tabular`

Status: `package`

Class: `ModelToken`

File: `graph_Time_series/token_blocks/tabular_models.py`

Purpose: LightGBM tabular regressor for multi-horizon forecasting.

Requires:

- `Port(sem="features", axes=("sample", "feature"), space="current", multiple=True, coerce=True)`

Reads:

- legacy `model_input` if present and tabular;
- else `state.feature_bundle()`;
- `current_target`.

Writes:

- `prediction_stack[-1]`
- `metadata["lightgbm_tabular"]`

Hyperparameters:

- `n_estimators=200`
- `learning_rate=0.05`
- `num_leaves=31`
- `max_depth=-1`
- `min_child_samples=20`
- `subsample=0.9`
- `colsample_bytree=0.9`
- `seed=0`
- `n_jobs=-1`

Implementation note:

- imports `lightgbm.LGBMRegressor` lazily;
- wraps it in `sklearn.multioutput.MultiOutputRegressor`.

Common next tokens:

- `STOP`

### `PeriodPhaseOneHot`

Status: `package`

Class: `FeatureToken`

File: `graph_Time_series/token_blocks/periodic.py`

Purpose: one-hot encode the point position inside the selected period.

Reads:

- `raw_history`
- `metadata["period"]`

Writes:

- `historical_features["history_period_phase_id"]`
- optional `historical_features["history_period_phase_onehot"]`
- `future_features["future_period_phase_id"]`
- `future_features["future_period_phase_onehot"]`
- `metadata["period_phase_onehot"]`
- `flags["period_phase_encoded"]`

Hyperparameters:

- `anchor="fold"` or `"history_start"`
- `include_history_onehot="auto"`
- `max_history_onehot_cells=20_000_000`
- `max_future_onehot_cells=50_000_000`
- `dtype=np.float32`

State effect:

```text
period + positions -> phase ids
phase ids -> horizon-side one-hot known feature
optional history-side one-hot feature
```

Common next tokens:

- `PeriodFold`

## FLAIR Tokens

These tokens are package tokens but opt-in through `register_flair_tokens`.
They expose the FLAIR decomposition as inspectable state instead of hiding it
inside one monolithic model.

### `FlairPreprocess`

Status: `package`

Class: `CleaningToken`

Purpose: interpolate finite values and shift each series positive.

Reads:

- `raw_history`

Writes:

- `historical_features["flair_history"]`
- `metadata["flair_shift"]`
- positivity-shift transform
- `flags["flair_preprocessed"]`

Hyperparameters:

- `eps=1e-6`

Next:

- `PeriodSelection`

### `PeriodSelection`

Status: `package`

Class: `FeatureToken`

Purpose: select a global FLAIR period using candidate period BIC/SVD scores.

Reads:

- `flair_history`

Writes:

- `metadata["period"]`
- `metadata["period_scores"]`
- `metadata["flair_secondary_periods"]`
- `flags["period_selected"]`

Hyperparameters:

- `freq="H"`
- `max_series=8`
- `min_complete=3`

Next:

- `PeriodPhaseOneHot`
- `PeriodFold`

### `PeriodFold`

Status: `package`

Class: `FeatureToken`

Purpose: fold preprocessed history into phase-by-period matrices.

Reads:

- `flair_history`
- `period`

Writes:

- `historical_features["period_matrix"]`
- `historical_features["level_series"]`
- compatibility aliases `flair_period_matrix`, `flair_level_raw`
- `metadata["n_complete_periods"]`

Hyperparameters:

- no constructor hyperparameters

Next:

- `LevelShrinkage`
- `ShapeLevel`

### `LevelShrinkage`

Status: `package`

Class: `FeatureToken`

Purpose: denoise period totals using a rank-1 energy shrinkage estimate.

Reads:

- `period_matrix`
- `level_series`

Writes:

- `historical_features["flair_level_denoised"]`
- `metadata["flair_level_shrinkage"]`

Hyperparameters:

- `min_factor=0.05`

Next:

- `ShapeLevel`
- `gb_level_forecast`

### `ShapeLevel`

Status: `package`

Class: `FeatureToken`

Purpose: estimate within-period shape proportions from recent complete periods.

Reads:

- `period_matrix`
- `level_series`

Writes:

- `historical_features["shape_vector"]`
- `historical_features["flair_shape"]`
- `historical_features["flair_shape_history"]`
- `metadata["shape_k"]`

Hyperparameters:

- `shape_k=2`

Next:

- `SecondaryLevelSeasonality`
- `gb_level_forecast`

### `SecondaryLevelSeasonality`

Status: `package`

Class: `FeatureToken`

Purpose: estimate secondary seasonality on the compressed level series.

Reads:

- `level_series`
- `period`

Writes:

- `historical_features["flair_shape2"]`
- `historical_features["flair_level_work"]`
- `features["flair_level_future_shape2"]`
- `metadata["flair_cross_period"]`
- `metadata["flair_cross_periods"]`

Hyperparameters:

- `min_complete=2`

Next:

- `LevelBoxCoxCenter`

### `LevelBoxCoxCenter`

Status: `package`

Class: `FeatureToken`

Purpose: apply per-series Box-Cox transform to FLAIR levels and center at the
latest level.

Reads:

- `flair_level_work`

Writes:

- `historical_features["flair_level_bc"]`
- `historical_features["flair_level_innov"]`
- `metadata["flair_boxcox"]`

Hyperparameters:

- `n_lambda_grid=21`
- `eps=1e-6`

Next:

- `FlairRidgeLevel`

### `FlairRidgeLevel`

Status: `package`

Class: `ModelToken`

Purpose: forecast compressed FLAIR levels with soft-averaged Ridge, expand
through shape, and push a point forecast.

Reads:

- `flair_level_innov`
- `shape_vector`
- `flair_boxcox`
- `period`
- optional secondary shape data

Writes:

- `prediction_stack[-1]`
- `features["flair_point_forecast"]`
- `features["flair_level_point"]`
- `features["flair_level_innov_point"]`
- ridge diagnostics
- `metadata["FlairRidgeLevel"]`

Hyperparameters:

- `alpha_log_min=-6.0`
- `alpha_log_max=3.0`
- `n_alphas=25`
- `show_progress=True`
- `progress_min_samples=32`

State effect:

```text
level innovations -> Ridge forecast -> inverse Box-Cox -> shape expansion
```

Next:

- `FlairSamplePaths`
- `STOP`

### `FlairSamplePaths`

Status: `package`

Class: `FeatureToken`

Purpose: generate FLAIR-style stochastic sample paths around the point forecast.

Reads:

- `flair_level_innov`
- `shape_vector`
- `period_matrix`
- `level_series`
- `flair_ridge_beta`
- `flair_ridge_phi`
- `flair_ridge_residuals`

Writes:

- `features["flair_samples"]`
- `features["flair_sample_mean"]`
- `metadata["FlairSamplePaths"]`

Hyperparameters:

- `n_paths=100`
- `seed=0`
- `phase_noise_scale=1.0`
- `clip_to_history=True`
- `max_sample_values=5_000_000`
- `show_progress=True`
- `progress_min_samples=16`

Next:

- `STOP`

## Versatile Tokens

These tokens demonstrate the signal-board design. They are opt-in through
`register_versatile_tokens`.

### `DayOfWeekFeature`

Status: `package`

Class: `FeatureToken`

Purpose: add cyclical calendar-like covariates using position modulo a period.

Reads:

- `raw_history`

Writes:

- `historical_features["calendar_features"]`
- `future_features["future_calendar_features"]`
- `flags["calendar_encoded"]`

Provides:

- history `features`
- future `features`

Hyperparameters:

- `period=7` in the class default
- `period=24` in `register_versatile_tokens`

Next:

- `FourierFeatures`
- `versatile_rf`
- `versatile_gb`

### `versatile_rf`

Status: `package`

Class: `ModelToken`

Purpose: RandomForestRegressor over any auto-built feature bundle.

Requires:

- history `features(sample, feature)` in current space, with coercion enabled.

Writes:

- `prediction_stack[-1]`
- `metadata["versatile_rf"]`

Hyperparameters:

- `n_estimators=80`
- `max_depth=10`
- `seed=0`
- `select_tags=()`

Next:

- `versatile_rf`
- `versatile_gb`
- `STOP`

### `versatile_gb`

Status: `package`

Class: `ModelToken`

Purpose: sklearn `HistGradientBoostingRegressor` over any auto-built feature
bundle.

Requires:

- history `features(sample, feature)` in current space, with coercion enabled.

Writes:

- `prediction_stack[-1]`
- `metadata["versatile_gb"]`

Hyperparameters:

- `max_iter=150`
- `learning_rate=0.06`
- `max_depth=None`
- `seed=0`
- `select_tags=()`

Next:

- `versatile_rf`
- `versatile_gb`
- `STOP`

### `gb_level_forecast`

Status: `package`

Class: `ModelToken`

Purpose: FLAIR sub-step replacement that forecasts period levels with gradient
boosting and expands through `shape_vector`.

Requires:

- `level(sample, period_idx)`
- `shape(sample, phase)`

Reads:

- `level_series`
- `shape_vector`
- `period`

Writes:

- `prediction_stack[-1]`
- `features["gb_level_point"]`
- `metadata["gb_level_forecast"]`

Hyperparameters:

- `n_lags=3`
- `max_iter=120`
- `learning_rate=0.08`
- `seed=0`

Next:

- `STOP`

## Common Sequences

Default package:

```text
ZNormalization -> kernel_rbf -> STOP
MeanAbsScaling -> FourierFeatures -> rf_tabular -> STOP
MeanAbsScaling -> FourierFeatures -> lightgbm_tabular -> STOP
ZNormalization -> FourierFeatures -> kernel_rbf -> STOP
```

Residual stacking:

```text
ZNormalization -> kernel_rbf -> kernel_rbf -> STOP
ZNormalization -> FourierFeatures -> versatile_rf -> versatile_gb -> STOP
```

FLAIR:

```text
FlairPreprocess
-> PeriodSelection
-> PeriodPhaseOneHot
-> PeriodFold
-> LevelShrinkage
-> ShapeLevel
-> SecondaryLevelSeasonality
-> LevelBoxCoxCenter
-> FlairRidgeLevel
-> STOP
```

FLAIR sample paths:

```text
FlairPreprocess
-> PeriodSelection
-> PeriodFold
-> ShapeLevel
-> SecondaryLevelSeasonality
-> LevelBoxCoxCenter
-> FlairRidgeLevel
-> FlairSamplePaths
-> STOP
```

FLAIR level-model swap:

```text
FlairPreprocess
-> PeriodSelection
-> PeriodFold
-> LevelShrinkage
-> ShapeLevel
-> gb_level_forecast
-> STOP
```

Calendar plus tabular:

```text
ZNormalization -> DayOfWeekFeature -> FourierFeatures -> versatile_rf -> STOP
```

## Notebook-Only / Prototype Space

The notebook remains the place to test ideas before promoting them into
package tokens. Useful directions, not implemented as current package tokens:

| Direction | Notes |
| --- | --- |
| STL decomposition | Decompose trend/seasonal/residual, then publish signals |
| Lag/rolling features | Create tabular history summaries for RF/LightGBM |
| Residual feature tokens | Let later models see previous residual structure |
| Sequence-native models | Use ports for multi-channel sequence tensors |
| Known-future covariates | Holidays, timestamps, external calendar data |
| Selector tokens | Optional feature-bundle narrowing by tags |

Any promoted token should update this file, the README, and the graph catalog.
