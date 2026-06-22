# Graph Time Series

Small experimental framework for composing time-series forecasting pipelines
from inspectable tokens.

The current architecture is centered on a broad `State` plus a semantic
Signal board. Tokens can still write to the older dictionaries
(`historical_features`, `future_features`, `features`, `metadata`, `flags`),
but those writes are mirrored into typed `Signal` objects. Models can then
consume the auto-built feature bundle instead of depending on one hard-coded
feature name.

The practical result:

- scalers, feature tokens, FLAIR-style decomposition tokens, and model tokens
  can be recombined more freely;
- tabular models no longer need explicit binder/glue tokens in normal use;
- residual stacking remains supported, so multiple model tokens can fit the
  remaining target after earlier predictions;
- the notebook can continue to inspect familiar state dictionaries while the
  package moves toward semantic ports.

For the token-by-token reference, see `TOKENS.md`.
For the current signal-board architecture, see `SIGNAL_ARCHITECTURE.md`.
For graph views of registered token transitions, see `token_graph_catalog/`.

## Repository Layout

```text
graph_Time_series/
|-- README.md
|-- TOKENS.md
|-- SIGNAL_ARCHITECTURE.md
|-- ARCHITECTURE_PROPOSAL.md
|-- check_list
|-- graph_Time_series/
|   |-- __init__.py
|   |-- artifacts.py
|   |-- grammar.py
|   |-- pipeline_ast.py
|   |-- signal.py
|   |-- state.py
|   |-- token.py
|   `-- token_blocks/
|       |-- __init__.py
|       |-- bindings.py
|       |-- flair.py
|       |-- fourier.py
|       |-- kernel_rbf.py
|       |-- normalization.py
|       |-- periodic.py
|       |-- tabular_models.py
|       `-- versatile.py
|-- examples/
|   `-- test_token_sequence_colab.ipynb
|-- tests/
|   `-- test_token_combinations.py
`-- token_graph_catalog/
    |-- README.md
    |-- token_catalog.json
    |-- render_token_graph.py
    `-- outputs/
```

## Core Concepts

### State

`State` is the runtime contract between tokens. It stores:

- immutable inputs: `original_history`, `original_future`;
- compatibility stores: `features`, `historical_features`, `future_features`;
- metadata and flags;
- artifact metadata;
- `board`, a list of typed `Signal` objects emitted by token writes;
- `current_space` and registered transforms;
- a prediction stack and the current residual target.

The residual invariant is still:

```text
current_target = active_target_base - sum(prediction_stack)
```

Any model token that calls `state.push_prediction(...)` fits the current
residual and makes the next model fit what remains. `get_final_prediction()`
decodes the cumulative prediction back to the original scale through the
registered inverse transforms.

### Signal Board

`graph_Time_series/signal.py` defines the semantic layer:

- `Signal`: a value with `sem`, `axes`, `alignment`, `space`, source, tags, and
  metadata.
- `Port`: a declarative requirement/provision contract for tokens.
- `Space` and `Transform`: the transform lineage.
- `build_feature_bundle(...)`: coercion of sample-leading signals into a single
  tabular matrix.

Tokens should couple on meaning, not on names. For example:

```text
scaled_history     -> Signal(sem="series", axes=("sample", "time"))
fourier_features   -> Signal(sem="features", axes=("sample", "feature"))
shape_vector       -> Signal(sem="shape", axes=("sample", "phase"))
level_series       -> Signal(sem="level", axes=("sample", "period_idx"))
model prediction   -> Signal(sem="forecast", axes=("sample", "horizon"))
```

### Feature Bundle

The normal model input path is now:

```text
state signals -> state.feature_bundle() -> model X matrix
```

`feature_bundle()` gathers finite, sample-aligned signals and flattens them into
`(n_samples, n_features)`. It records block metadata so the notebook can still
inspect which pieces were used.

The old `historical_features["model_input"]` path is still honored for backward
compatibility, but the package no longer ships active binder/glue tokens.
`token_blocks/bindings.py` is intentionally empty except for the removal note.

### Tokens

The base classes live in `token.py`:

| Class | Purpose |
| --- | --- |
| `CleaningToken` | Pre-model data cleanup or context selection |
| `FeatureToken` | Creates reusable features or decomposition artifacts |
| `TransformToken` | Changes target/feature scale and registers inverse transforms |
| `ModelToken` | Fits on `state.current_target` and pushes predictions |
| `ControlToken` | Sequence controls such as `STOP` |

Tokens expose legacy `reads` / `writes` for logging and may also expose semantic
`requires` / `provides` ports. `can_apply(state)` checks usage caps, legacy
dependencies, port requirements, and token-specific conditions.

## Token Registries

The package exposes several registry helpers in `token_blocks/__init__.py`.

### Default Registry

```python
from graph_Time_series.grammar import Grammar
from graph_Time_series.token_blocks import register_default_tokens

grammar = register_default_tokens(Grammar())
```

Registered tokens:

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
parrot_dataset
```

Typical binder-free chains:

```text
LinearFill -> ZNormalization -> kernel_rbf -> STOP
ZNormalization -> kernel_rbf -> STOP
ZNormalization -> parrot -> kernel_rbf -> STOP
ZNormalization -> kernel_rbf -> parrot -> STOP
ZNormalization -> parrot_dataset -> kernel_rbf -> STOP
MeanAbsScaling -> FourierFeatures -> rf_tabular -> STOP
MeanAbsScaling -> FourierFeatures -> lightgbm_tabular -> STOP
ZNormalization -> FourierFeatures -> kernel_rbf -> kernel_rbf -> STOP
```

### FLAIR Registry

```python
from graph_Time_series.token_blocks import register_flair_tokens

grammar = register_flair_tokens(register_default_tokens(Grammar()))
```

Adds:

```text
FlairPreprocess
PeriodDetect
PeriodDetectSpectral
PeriodDetectBIC
PeriodPhaseOneHot
SeasonalFold
level_shape_ridge
FlairRidgeLevel
FlairSamplePaths
```

Typical FLAIR point forecast:

```text
LinearFill
-> PeriodDetect
-> PeriodPhaseOneHot
-> SeasonalFold
-> level_shape_ridge
-> STOP
```

Verbose FLAIR inspection path:

```text
LinearFill
-> PeriodDetect
-> PeriodPhaseOneHot
-> SeasonalFold
-> FlairRidgeLevel
-> STOP
```

`FlairSamplePaths` can follow `FlairRidgeLevel` to generate stochastic sample
paths.

### Versatile Registry

```python
from graph_Time_series.token_blocks import register_versatile_tokens

grammar = register_versatile_tokens(register_default_tokens(Grammar()))
```

Adds:

```text
versatile_gb
```

These are demonstration tokens for the signal-board architecture. The model
tokens use `state.feature_bundle()` and can pick up whatever compatible feature
signals exist.

Example:

```text
ZNormalization -> FourierFeatures -> versatile_gb -> STOP
```

### FLAIR GB Swap

```python
from graph_Time_series.token_blocks import register_flair_gb_swap

grammar = register_flair_gb_swap(register_flair_tokens(register_default_tokens(Grammar())))
```

Adds:

```text
gb_level_forecast
```

This is a drop-in model-side experiment after `SeasonalFold`: it predicts period
levels with gradient boosting and expands them through the learned shape.

### Notebook-Only / Unfinished Tokens

These tokens are present in the working tree but are not stable package tokens
yet. They are loaded directly by `examples/pipeline_inspect.py` for
troubleshooting and should be promoted deliberately before they are included in
the main registry or graph catalog:

```text
window_kernel
affine_fold
affine_forecast
```

Current flags:

- `window_kernel`: cross-series window analog model; graph edges and input
  priority still need final cleanup.
- `affine_fold`: affine seasonal decomposition prototype; helper-only for now.
- `affine_forecast`: forecast head for `affine_fold`; should move with it if
  promoted.

## Implemented Package Tokens

### Cleaning

`LinearFill`

- fills NaN/inf history values by linear interpolation;
- writes `clean_history`;
- refreshes the active `raw_history` view used by later tokens.

`ForwardFill`

- fills NaN/inf history values by carrying the latest finite value forward;
- leading gaps use the first finite value;
- writes the same minimal outputs as `LinearFill`.

### Transforms

`ZNormalization`

- reads `raw_history`;
- writes `scaled_history`;
- transforms the active target by per-series mean/std;
- registers the inverse transform.

`MeanAbsScaling`

- reads `raw_history`;
- writes `scaled_history`;
- scales history and target by per-series mean absolute history value;
- registers the inverse transform.

### Features And Decomposition

`FourierFeatures`

- creates fixed-width FFT features from `scaled_history` or raw history;
- writes `fourier_features`;
- useful before `rf_tabular`, `lightgbm_tabular`, `kernel_rbf`, and versatile
  models.

`PeriodPhaseOneHot`

- requires `metadata["period"]`;
- writes history/future phase IDs and future one-hot period positions;
- optionally writes history one-hot when the tensor is below its configured
  size guard.

FLAIR feature tokens:

- `LinearFill` or compatibility alias `FlairPreprocess`
- `PeriodDetect`
- `PeriodDetectSpectral`
- `PeriodDetectBIC`
- `SeasonalFold`
- `FlairSamplePaths`

### Models

`kernel_rbf`

- KernelRidge with RBF kernel;
- uses explicit legacy `model_input` if present;
- otherwise uses `state.feature_bundle()`;
- falls back to `scaled_history` for older direct usage;
- fits leave-one-out predictions inside the current state.

`rf_tabular`

- `RandomForestRegressor` over the active feature bundle;
- supports multi-horizon targets directly.

`lightgbm_tabular`

- LightGBM `LGBMRegressor` wrapped in `MultiOutputRegressor`;
- imports LightGBM lazily, so `lightgbm` must be installed only when this token
  is used.

`parrot`

- within-series analog / nearest-neighbour forecaster;
- reads the active time series view: `scaled_history`, then `clean_history`,
  then active `raw_history`;
- pushes into the normal prediction stack, so later models fit the residual;
- can also follow another model and add its own residual prediction.

`parrot_dataset`

- cross-series analog / nearest-neighbour forecaster over candidate windows;
- uses bounded candidate sampling for scalability;
- pushes into the normal prediction stack.

`level_shape_ridge`

- compact period-level predictor;
- reads existing level/shape state;
- handles Box-Cox positivity internally;
- does not write separate Box-Cox feature tokens, ridge coefficients, or level forecasts into
  `State`;
- pushes only the final horizon forecast through the normal prediction stack.

`FlairRidgeLevel`

- FLAIR-style soft-averaged Ridge with internal Box-Cox level centering;
- expands through the learned shape and pushes a horizon forecast.

`versatile_gb`

- experimental model token that declares semantic feature ports and fits on the
  auto-built bundle;
- useful for testing whether new feature tokens combine cleanly.

`gb_level_forecast`

- experimental FLAIR sub-step replacement;
- predicts period-level values with gradient boosting and expands via shape.

## Pipeline DSL

The optional Lark parser lives in `pipeline_ast.py`. It parses simple linear
pipelines:

```python
from graph_Time_series.pipeline_ast import parse_pipeline

ast = parse_pipeline("ZNormalization -> FourierFeatures -> rf_tabular -> STOP")
```

The DSL is intentionally thin: it expresses order, while `Grammar` and
`Token.can_apply(...)` still decide whether a sequence is valid for a given
state.

## Notebook

`examples/test_token_sequence_colab.ipynb` is the main exploration notebook. It
is used for:

- loading datasets and checking holdout-only comparisons;
- testing token sequences interactively;
- comparing kernel, RF, and LightGBM paths;
- exploring prototype ideas before promoting them into package tokens.

Any long loop in the notebook should use `tqdm` from now on.

## Validation

Useful smoke checks:

```bash
python -m py_compile graph_Time_series/token_blocks/imputation.py graph_Time_series/token_blocks/flair.py graph_Time_series/token_blocks/seasonal.py graph_Time_series/token_blocks/normalization.py graph_Time_series/token_blocks/__init__.py examples/pipeline_inspect.py examples/list_all_sequences.py
python -c "from graph_Time_series.grammar import Grammar; from graph_Time_series.token_blocks import register_default_tokens, register_flair_tokens, register_versatile_tokens, register_flair_gb_swap, register_seasonal_tokens; g=Grammar(); [globals()[name](g) for name in ['register_default_tokens','register_flair_tokens','register_versatile_tokens','register_flair_gb_swap','register_seasonal_tokens']]; print(g)"
```

Current live registry:

```text
Grammar(23 tokens, 123 edges)
```

In this Windows workspace, the `python` command may point to the WindowsApps
launcher. The working interpreter used during review was:

```text
C:\Users\chahine\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

## Adding Tokens

Do not add or modify tokens casually. The working convention is:

1. propose the token and get explicit approval;
2. define the state effect first: reads, writes, signals, alignment, and space;
3. add constructor hyperparameters deliberately;
4. expose semantic `requires` / `provides` when the token should combine with
   other families;
5. add a focused test or notebook sequence showing the expected combination.

Open-ended directions kept for later:

- STL decomposition tokens;
- lag/rolling/tabular feature generators;
- residual-specific model tokens and residual selectors;
- sequence-native models with multi-channel ports;
- known-future covariates such as holidays or real timestamps;
- search policies that use port compatibility to control graph size.
