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
LinearFill
ForwardFill
ZNormalization
MeanAbsScaling
FourierFeatures
kernel_rbf
rf_tabular
lightgbm_tabular
parrot
```

### FLAIR

Registered by `register_flair_tokens(grammar)`:

```text
FlairPreprocess
PeriodDetect
PeriodDetectSpectral
PeriodDetectBIC
PeriodPhaseOneHot
SeasonalFold
level_shape_ridge
LevelBoxCoxCenter
FlairRidgeLevel
FlairSamplePaths
```

### Versatile

Registered by `register_versatile_tokens(grammar)`:

```text
versatile_gb
```

### FLAIR GB Swap

Registered by `register_flair_gb_swap(grammar)`:

```text
gb_level_forecast
```

### Seasonal

Registered by `register_seasonal_tokens(grammar)`:

```text
PeriodDetect
PeriodDetectSpectral
PeriodDetectBIC
SeasonalFeatures
step_regression
```

With all registries enabled:

```text
Grammar(23 tokens, 112 edges)
```

## Package Tokens

### `LinearFill`

Status: `package`

Class: `CleaningToken`

File: `graph_Time_series/token_blocks/imputation.py`

Purpose: fill NaN/inf history values by linear interpolation.

Reads:

- `raw_history`

Writes:

- active `features["raw_history"]`
- `historical_features["clean_history"]`
- `flags["history_filled"]`

Hyperparameters:

- no constructor hyperparameters

State effect:

```text
raw_history with gaps -> interpolated raw_history + clean_history
```

Common next tokens:

- `ZNormalization`
- `MeanAbsScaling`
- `FourierFeatures`
- `kernel_rbf`
- `rf_tabular`
- `lightgbm_tabular`
- `PeriodDetect`

### `ForwardFill`

Status: `package`

Class: `CleaningToken`

File: `graph_Time_series/token_blocks/imputation.py`

Purpose: fill NaN/inf history values by carrying the latest finite value forward.

Reads:

- `raw_history`

Writes:

- active `features["raw_history"]`
- `historical_features["clean_history"]`
- `flags["history_filled"]`

Hyperparameters:

- no constructor hyperparameters

State effect:

```text
leading gaps -> first finite value
later gaps -> previous finite value
all-missing rows -> zeros
```

Common next tokens:

- `ZNormalization`
- `MeanAbsScaling`
- `FourierFeatures`
- `kernel_rbf`
- `rf_tabular`
- `lightgbm_tabular`
- `PeriodDetect`

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
- `versatile_gb`

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
- `parrot`
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

- `parrot`
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

- `parrot`
- `STOP`

### `parrot`

Status: `package`

Class: `ModelToken`

File: `graph_Time_series/token_blocks/parrot.py`

Purpose: within-series analog / nearest-neighbour forecaster.

Reads:

- preferred `scaled_history`
- fallback `clean_history`
- fallback active `features["raw_history"]`
- `current_target` through the residual stack

Writes:

- `prediction_stack[-1]`
- updated `current_target` / `current_residual`

Hyperparameters:

- `source_feature=None` for automatic source selection
- fixed analog window length: forecast horizon
- fixed neighbours: `1`
- match score: absolute Pearson correlation

State effect:

```text
active history sequence -> best past analog continuation
prediction -> state.push_prediction(...)
current_target becomes residual for later models
```

Common next tokens:

- `kernel_rbf`
- `rf_tabular`
- `lightgbm_tabular`
- `versatile_gb`
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

- `SeasonalFold`

## FLAIR Tokens

These tokens are package tokens but opt-in through `register_flair_tokens`.
They expose the FLAIR decomposition as inspectable state instead of hiding it
inside one monolithic model.

### `FlairPreprocess`

Status: `package`

Class: `CleaningToken`

Purpose: backward-compatible alias for `LinearFill`.

Reads:

- `raw_history`

Writes:

- active `features["raw_history"]`
- `historical_features["clean_history"]`
- `flags["history_filled"]`

Hyperparameters:

- no constructor hyperparameters

State effect:

```text
same as LinearFill; no positivity shift
```

Next:

- `PeriodDetect`

### `PeriodDetect`

Status: `package`

Class: `FeatureToken`

Purpose: default FLAIR-style BIC period detector.

Reads:

- preferred `clean_history`
- fallback `flair_history`, `scaled_history`, then `raw_history`

Writes:

- `metadata["periods"]`
- compatibility scalar `metadata["period"]`
- `metadata["period_scores"]`
- `metadata["period_baseline_bic"]`
- `metadata["period_selector"] = "bic"`
- `flags["periods_detected"]`

Hyperparameters:

- `freq="H"`
- `max_series=8`
- `min_complete=3`
- `source_feature="scaled_history"`
- `margin=0.15`
- `max_periods=4`

Next:

- `PeriodPhaseOneHot`
- `SeasonalFold`

### `PeriodDetectSpectral`

Status: `package`

Class: `FeatureToken`

Purpose: detect a period chain by periodogram power.

Reads:

- preferred `clean_history`
- fallback `flair_history`, `scaled_history`, then `raw_history`

Writes:

- `metadata["periods"]`
- compatibility scalar `metadata["period"]`
- `metadata["period_power"]`
- `metadata["period_selector"] = "periodogram"`
- `flags["periods_detected"]`

Hyperparameters:

- `freq="H"`
- `max_series=8`
- `min_complete=3`
- `source_feature="scaled_history"`
- `power_threshold=0.05`
- `band=1`
- `max_periods=4`

Next:

- `PeriodPhaseOneHot`
- `SeasonalFold`
- `SeasonalFeatures`

### `PeriodDetectBIC`

Status: `package`

Class: `FeatureToken`

Purpose: detect a nested period chain by rank-1 fold/BIC score.

Reads:

- preferred `clean_history`
- fallback `flair_history`, `scaled_history`, then `raw_history`

Writes:

- `metadata["periods"]`
- compatibility scalar `metadata["period"]`
- `metadata["period_scores"]`
- `metadata["period_baseline_bic"]`
- `metadata["period_selector"] = "bic"`
- `flags["periods_detected"]`

Hyperparameters:

- `freq="H"`
- `max_series=8`
- `min_complete=3`
- `source_feature="scaled_history"`
- `margin=0.15`
- `max_periods=4`

Next:

- `PeriodPhaseOneHot`
- `SeasonalFold`

### `SeasonalFold`

Status: `package`

Class: `FeatureToken`

Purpose: one seasonal fold per call. Each `apply` does **exactly one** fold (it
is NOT internally recursive); calling the token again via the grammar self-loop
pops the **next** period from `PeriodDetect`'s list and folds the running
amplitude, like a queue. It replaces the former `PeriodFold` -> `ShapeLevel` ->
`SecondaryLevelSeasonality` sequence with a single primitive,
`fold(series, p) -> shape (proportions, sum to 1) + per-cycle amplitude`.

- the first call folds `clean_history` when present, otherwise active
  `raw_history`, by the dominant period `periods[0]`;
- each later call folds the amplitude left by the previous call by the next
  nested ratio `periods[d] // P_prev`, accumulating the separable shape in one
  multiply.

Called once it is a plain seasonal decomposition; called again it peels the
next nested period. After d+1 folds the total period is `P` and the
decomposition collapses to the standard head interface:

- `level_series` = per-P-cycle amplitude (sum over the P phases),
- `shape_vector` = SEPARABLE shape over P phases = outer product of the
  per-level shapes, `eff[t] = prod_d shape_d[(t // periods[d-1]) % fold_p_d]`,
- `metadata["period"] = P`.

So `level_shape_ridge`, `LevelBoxCoxCenter -> FlairRidgeLevel` and
`gb_level_forecast` (which reconstruct `level[t//P] * shape[t%P]`) work
unchanged -- they forecast only the innermost amplitude and re-drape the
separable shape. The old `future_shape2` / `flair_cross_period` secondary
machinery is no longer used (seeded to identity for compatibility).

Reads:

- `clean_history` when present, otherwise active `raw_history`
- `param:periods` (required port; from `PeriodDetect`)

Writes:

- `historical_features["period_matrix"]` (history folded by the total period P),
  `["level_series"]`, `["shape_vector"]`, `flair_level_work`,
  compatibility aliases `flair_period_matrix`, `flair_level_raw`, `flair_shape`
- `metadata["period"]` (= P), `["n_complete_periods"]`, `["shape_k"]`,
  `["seasonal_fold_depth"]`, `["seasonal_fold_periods"]`
- when `shrink_level=True`: `historical_features["flair_level_denoised"]`,
  `metadata["flair_level_shrinkage"]`

Hyperparameters:

- `shrink_level=True`, `min_factor=0.05` (rank-1 SVD level shrinkage on the
  per-P-cycle amplitude)
- `shape_k=2` (cycles averaged into each frozen shape)
- `min_complete=2` (minimum complete P-cycles required to fold)
- `max_uses=4`

Nesting requirement: a further fold is possible only when the next period is an
integer multiple of the previous one (`periods[d] % periods[d-1] == 0`) with
enough complete cycles. With a single detected period exactly one fold applies
(one seasonality); a 2-fold sequence needs two nested periods.

Next:

- `SeasonalFold` (fold the next nested period)
- `level_shape_ridge`
- `LevelBoxCoxCenter`
- `gb_level_forecast`

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

- `versatile_gb`
- `parrot`
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

- `parrot`
- `STOP`

## Common Sequences

Default package:

```text
ZNormalization -> kernel_rbf -> STOP
ZNormalization -> parrot -> kernel_rbf -> STOP
ZNormalization -> kernel_rbf -> parrot -> STOP
MeanAbsScaling -> FourierFeatures -> rf_tabular -> STOP
MeanAbsScaling -> FourierFeatures -> lightgbm_tabular -> STOP
ZNormalization -> FourierFeatures -> kernel_rbf -> STOP
```

Residual stacking:

```text
ZNormalization -> kernel_rbf -> kernel_rbf -> STOP
ZNormalization -> FourierFeatures -> versatile_gb -> versatile_gb -> STOP
```

FLAIR:

```text
LinearFill
-> PeriodDetect
-> SeasonalFold
-> level_shape_ridge
-> STOP
```

FLAIR verbose inspection path:

```text
LinearFill
-> PeriodDetect
-> PeriodPhaseOneHot
-> SeasonalFold
-> LevelBoxCoxCenter
-> FlairRidgeLevel
-> STOP
```

FLAIR sample paths:

```text
LinearFill
-> PeriodDetect
-> SeasonalFold
-> LevelBoxCoxCenter
-> FlairRidgeLevel
-> FlairSamplePaths
-> STOP
```

FLAIR level-model swap:

```text
LinearFill
-> PeriodDetect
-> SeasonalFold
-> gb_level_forecast
-> STOP
```

Seasonal plus tabular:

```text
ZNormalization -> PeriodDetect -> SeasonalFeatures -> rf_tabular -> STOP
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
