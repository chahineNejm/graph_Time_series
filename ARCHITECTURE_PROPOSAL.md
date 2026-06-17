# Architecture Notes: Implemented Signal Layer And Next Steps

This file is no longer a pure proposal. The project now implements the main
signal-board idea as a backward-compatible layer on top of the existing
`State`. This document records what exists today and which parts are still
future architecture.

## Goal

Support many valid token sequences without forcing every model to know every
possible upstream artifact name.

The current answer is:

```text
tokens write broad state
State emits typed Signals
models consume semantic feature bundles
residual stacking composes model outputs
```

Tokens should couple on meaning:

```text
series, features, level, shape, forecast, samples, param:*
```

and on axes:

```text
sample, time, horizon, feature, phase, period_idx, path
```

instead of coupling on one artifact name such as `scaled_history`.

## Implemented Today

### Signal

`graph_Time_series/signal.py` defines:

```python
Signal(value, sem, axes, alignment, space, name, source, tags, meta)
```

Examples:

| Artifact | Signal meaning |
| --- | --- |
| `raw_history` | `sem="series"`, `axes=("sample", "time")`, raw space |
| `scaled_history` | `sem="series"`, `axes=("sample", "time")`, current space |
| `fourier_features` | `sem="features"`, `axes=("sample", "feature")` |
| `period_matrix` | `sem="series"`, `axes=("sample", "phase", "period_idx")` |
| `level_series` | `sem="level"`, `axes=("sample", "period_idx")` |
| `shape_vector` | `sem="shape"`, `axes=("sample", "phase")` |
| model output | `sem="forecast"`, `axes=("sample", "horizon")` |

### State Bridge

`State` still owns the old stores because they are useful for notebooks and
compatibility:

```text
features
historical_features
future_features
metadata
flags
```

The important new fields are:

```text
board
current_space
transforms
```

When a token registers an artifact or pushes a prediction, `State` emits a
`Signal`. This lets old-style and new-style tokens coexist.

### Ports

`Token` supports:

```python
requires = (...)
provides = (...)
```

The `Port` contract is currently used by the tabular and versatile models to
ask for feature-like data rather than one specific artifact.

### Auto Feature Bundle

`state.feature_bundle()` is implemented. It gathers feature-compatible signals,
coerces them to `(sample, feature)`, concatenates them, and returns block
metadata for inspection.

This is the replacement for package-level binder/glue tokens.

### Residual Models

The residual rule is unchanged:

```text
current_target = active_target_base - sum(prediction_stack)
```

Any model that calls `state.push_prediction(...)` can be followed by another
model that fits the remaining residual, provided the grammar and token
conditions allow it.

## Removed From Normal Operation

Package binder tokens such as `BindScaledHistory` and `BindAllSafeTabular` are
not part of the current runtime architecture. `token_blocks/bindings.py` is a
removal stub.

Models still check `historical_features["model_input"]` for backward
compatibility if an old notebook creates it manually, but new examples and docs
should use the signal-board bundle path.

## Current Registries

Default package registry:

```text
ZNormalization
MeanAbsScaling
FourierFeatures
kernel_rbf
rf_tabular
lightgbm_tabular
```

FLAIR opt-in registry:

```text
FlairPreprocess
PeriodDetect
PeriodPhaseOneHot
PeriodFold
ShapeLevel
SecondaryLevelSeasonality
level_shape_ridge
LevelBoxCoxCenter
FlairRidgeLevel
FlairSamplePaths
```

Versatile opt-in registry:

```text
versatile_gb
```

FLAIR swap opt-in registry:

```text
gb_level_forecast
```

Seasonal opt-in registry:

```text
PeriodDetect
SeasonalFeatures
step_regression
```

With all registries enabled:

```text
Grammar(20 tokens, 57 edges)
```

## Still Proposal / Next Architecture

### Derived Grammar

The grammar is still explicitly registered. The future version should derive
many edges from:

- token class order;
- `requires`;
- `provides`;
- adapter reachability;
- usage caps.

This would reduce hand wiring as token count grows.

### Stronger Board Immutability

`Signal` objects are frozen, but array values can still be mutable. Before heavy
MCTS or parallel rollout search, choose one policy:

- make signal arrays read-only;
- deep-copy array values on state branch;
- enforce token convention that signals are never mutated in place.

### More Explicit Axis Contracts

Axis inference currently works for the implemented adapters. Future structured
models should be stricter about:

- horizon-side known covariates;
- multi-channel sequence tensors;
- per-period matrices;
- sample-path tensors.

### Selector Tokens

The old binder idea may come back in a different form as selector tokens. A
selector would not create required plumbing; it would narrow or tag which
signals a model should consume.

Example future direction:

```text
FourierFeatures -> SelectTags(fourier) -> rf_tabular
```

That is different from the old architecture where a binder was required before
a model could run.

### More Token Families

Open possibilities:

- STL decomposition and residual tokens;
- lag, rolling-window, and calendar feature tokens;
- sequence-native models with multi-channel ports;
- known-future covariates;
- search policies that score combinations without exploding the graph.

## Migration Rule

For new tokens, prefer this order:

1. define the state effect and semantic meaning;
2. write outputs with `register_artifact(...)` so signals are emitted;
3. declare ports when the token needs semantic inputs;
4. rely on `state.feature_bundle()` for tabular models;
5. add a small combination test before widening the search.
