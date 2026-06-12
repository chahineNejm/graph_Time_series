# Signal-Board Architecture

This is the implemented architecture in the current checkout. It is additive
and backward compatible: older token code can still use the dictionary stores,
while newer code can query semantic `Signal` objects.

## What Changed

### `graph_Time_series/signal.py`

The signal module defines the common language between tokens:

- `Signal`: a value plus semantic metadata.
- `Space`: the active transform lineage.
- `Transform`: a named transform with inverse metadata.
- `Port`: a declarative token input/output contract.
- `build_feature_bundle(...)`: adapter-based flattening into tabular features.

Signals are identified by meaning:

```text
sem         what the value means: series, features, level, shape, forecast, samples, param:*
axes        dimensions such as sample, time, horizon, feature, phase, period_idx
alignment   history, future, static, or any
space       raw, current, or a transform lineage
tags        optional selectors such as fourier, flair, calendar
```

The important design choice is that `name` is provenance, not the primary
interface. Tokens should prefer semantic requirements over exact artifact names
when possible.

### `State`

`State` now has:

- the legacy stores: `features`, `historical_features`, `future_features`,
  `metadata`, `flags`;
- the artifact registry;
- `board: list[Signal]`;
- `current_space`;
- transform lineage;
- residual target and prediction stack.

Existing helpers such as `add_historical_feature(...)`,
`register_artifact(...)`, `register_transform(...)`, and `push_prediction(...)`
emit signals onto the board. This means old-style tokens populate the new
semantic layer without a rewrite.

New query helpers:

```python
state.query(...)
state.one(...)
state.has(...)
state.feature_bundle(...)
state.put_param(...)
```

`state.feature_bundle(...)` is the main bridge for tabular models. It gathers
finite sample-aligned signals and returns one matrix plus block metadata.

### `Token`

Tokens still support legacy `reads` / `writes`, but they may also declare:

```python
requires = (
    Port(sem="features", axes=("sample", "feature"),
         alignment="history", space="current", multiple=True, coerce=True),
)
```

`can_apply(state)` checks ports through the signal board. If a features port has
`coerce=True`, the feature bundle adapter path can satisfy it.

## Current Runtime Shape

The current implementation is not a total rewrite. It is a bridge:

```text
legacy dictionaries + artifact registry
        |
        v
Signal board
        |
        v
feature_bundle / semantic queries
        |
        v
model tokens and future token families
```

This is intentional. The notebook remains easy to debug, while token
interchangeability improves.

## Binder Status

Package binder/glue tokens have been removed from normal operation.

`token_blocks/bindings.py` is intentionally a removal stub. Models now consume
the signal-board auto-bundle directly:

```text
ZNormalization -> kernel_rbf -> STOP
MeanAbsScaling -> FourierFeatures -> rf_tabular -> STOP
MeanAbsScaling -> FourierFeatures -> lightgbm_tabular -> STOP
```

For backward compatibility, `kernel_rbf`, `rf_tabular`, and `lightgbm_tabular`
still honor `historical_features["model_input"]` if an older notebook creates
one manually. That path should be treated as legacy compatibility, not the
current package architecture.

## Registries

`register_default_tokens(grammar)`:

- `ZNormalization`
- `MeanAbsScaling`
- `FourierFeatures`
- `kernel_rbf`
- `rf_tabular`
- `lightgbm_tabular`

`register_flair_tokens(grammar)`:

- `FlairPreprocess`
- `PeriodSelection`
- `PeriodPhaseOneHot`
- `PeriodFold`
- `LevelShrinkage`
- `ShapeLevel`
- `SecondaryLevelSeasonality`
- `LevelBoxCoxCenter`
- `FlairRidgeLevel`
- `FlairSamplePaths`

`register_versatile_tokens(grammar)`:

- `DayOfWeekFeature`
- `versatile_rf`
- `versatile_gb`

`register_flair_gb_swap(grammar)`:

- `gb_level_forecast`

When all registries are enabled together, the grammar currently reports:

```text
Grammar(20 tokens, 58 edges)
```

## Demonstrated Combinations

`tests/test_token_combinations.py` runs synthetic data through:

- full FLAIR;
- FLAIR with `gb_level_forecast` replacing the Ridge level forecaster;
- binder-free RF and gradient boosting variants;
- package `rf_tabular` without binder tokens;
- calendar feature inserted before RF;
- `kernel_rbf` without binder tokens;
- time-bounded reachability enumeration.

Current expected output:

```text
7/7 named pipelines passed
completed pipelines: 41
failures: 0
RESULT: ALL GREEN
```

The test can emit sklearn/joblib warnings about worker configuration. Those
warnings are noisy but are not current correctness failures.

## Known Architectural Caveats

- `Signal` is frozen, but the stored value may be a mutable NumPy array. Current
  tokens usually allocate new arrays. For heavier search/MCTS, either treat
  signal arrays as immutable by convention or make board copies protect array
  values more strongly.
- The grammar is still hand-registered. Semantic ports now make derived graph
  edges possible later, but that derivation is not implemented yet.
- Axis inference is pragmatic. It is good enough for current adapters, but
  future strict sequence or horizon-native models may need more explicit axes.
- LightGBM is optional and imported lazily by `lightgbm_tabular`.
