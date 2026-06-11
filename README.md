# Graph Time Series

Small experimental framework for composing time-series forecasting pipelines from
inspectable tokens.

The current checkout is focused on a compact core:

- a shared `State` object that owns features, transforms, residuals, artifacts,
  and model-input bundles;
- token classes for transforms, feature/binding steps, models, and controls;
- a graph `Grammar` for valid token transitions;
- an optional Lark pipeline DSL for writing linear token chains;
- a Colab notebook used as the main playground for dataset loading,
  exploration, leakage-safe holdout scoring, and prototype tokens.

This README describes the code that is present in this repository now. Older
ideas such as large MCTS search, random forests, XGBoost, FFT encoders, and
other model families are kept as open extension directions rather than listed as
implemented blocks.

For a growing token-by-token index, see `TOKENS.md`.
For graph views of token parents and possible next tokens, see
`token_graph_catalog/`.

## Repository Layout

```text
graph_Time_series/
|-- README.md
|-- TOKENS.md
|-- check_list
|-- token_graph_catalog/
|   |-- README.md
|   |-- token_catalog.json
|   |-- render_token_graph.py
|   `-- outputs/
|-- graph_Time_series/
|   |-- __init__.py
|   |-- artifacts.py
|   |-- grammar.py
|   |-- pipeline_ast.py
|   |-- state.py
|   |-- token.py
|   `-- token_blocks/
|       |-- __init__.py
|       |-- bindings.py
|       |-- kernel_rbf.py
|       `-- normalization.py
`-- examples/
    `-- test_token_sequence_colab.ipynb
```

## Core Concepts

### State

`State` is the runtime contract between tokens. It stores:

- immutable input references: `original_history`, `original_future`;
- historical and future feature stores;
- legacy/general `features` for compatibility and notebook inspection;
- metadata and flags;
- an artifact registry describing values created by tokens;
- model-input bundles describing which artifacts a model is allowed to consume;
- target transforms and inverse transforms;
- a prediction stack and current residual target.

The important invariant is:

```text
current_target = active_target_base - sum(prediction_stack)
```

This means models can be chained: the first model predicts the transformed
target, the second model fits the residual, the third model fits the next
residual, and so on. `get_final_prediction()` decodes the cumulative prediction
back to the original data scale using registered inverse transforms.

### Artifacts And Bundles

Feature tokens are allowed to create broad state. They can leave many named
artifacts in `historical_features`, `future_features`, or `features`.

Binder tokens then decide what becomes the active model input:

```text
many artifacts in State -> one active InputBundle -> model_input
```

This keeps search/control manageable. Models do not need to guess from every
possible feature in the state; they read one explicit `model_input` bundle, plus
bundle metadata such as kind, shape, source token, and component artifacts.

### Tokens

The token base classes live in `token.py`:

| Class | Purpose |
| --- | --- |
| `CleaningToken` | Pre-model data cleanup or context selection |
| `FeatureToken` | Creates reusable features or binding decisions |
| `TransformToken` | Changes target/feature scale and registers inverses |
| `ModelToken` | Fits on `state.current_target` and pushes predictions |
| `ControlToken` | Sequence controls such as `STOP` |

Each token exposes:

- `name`
- `token_class`
- `reads`
- `writes`
- `can_apply(state)`
- `apply(state)`

`can_apply()` checks class/token usage caps, declared dependencies, and
token-specific conditions. `apply()` mutates or copies the state and records the
execution log.

## Implemented Package Tokens

### `ZNormalization`

File: `graph_Time_series/token_blocks/normalization.py`

Per-series z-normalization based on the history. It writes
`scaled_history`, transforms the active target into normalized space, and
registers an inverse transform so final predictions are decoded correctly.

Typical use:

```text
ZNormalization -> BindScaledHistory -> kernel_rbf
```

### `MeanAbsScaling`

File: `graph_Time_series/token_blocks/normalization.py`

Per-series scaling by the mean absolute history value, without centering. It
writes the same `scaled_history` contract as `ZNormalization`, registers an
inverse transform, and can be followed by the same bind/model tokens.

Typical use:

```text
MeanAbsScaling -> BindScaledHistory -> kernel_rbf
```

### `BindScaledHistory`

File: `graph_Time_series/token_blocks/bindings.py`

Binds `scaled_history` as the active `model_input`.

Bundle metadata:

```text
name = "scaled_history"
kind = "sequence_flat"
artifact_names = ("scaled_history",)
```

### `BindAllSafeTabular`

File: `graph_Time_series/token_blocks/bindings.py`

Flattens all finite historical artifacts, except an existing `model_input`, into
one tabular bundle. This is useful for quick experiments, but it can make the
feature space large, so more targeted binders are often preferable.

### `BindFeatureToken`

File: `graph_Time_series/token_blocks/bindings.py`

Generic binder for one named artifact. It is intended for creating configured
tokens such as `BindRawHistory`, `BindLevelSeries`, or `BindCalendarFeatures`.

### `StackFeatureBundleToken`

File: `graph_Time_series/token_blocks/bindings.py`

Generic binder that flattens and concatenates selected artifacts into one
tabular bundle. This supports controlled multi-input experiments without making
every model inspect the entire state.

### `PeriodPhaseOneHot`

File: `graph_Time_series/token_blocks/periodic.py`

Encodes positions inside a selected period. It expects `metadata["period"]` to
already exist, so in the current notebook it is used after `PeriodSelection`.
It writes compact history/future phase IDs, a future one-hot feature, and a
history one-hot feature when the tensor is small enough.

This token is exported from the package, but it is not part of the default
package grammar yet because the period-selection token still lives in the
notebook.

### `FourierFeatures`

File: `graph_Time_series/token_blocks/fourier.py`

Creates fixed-width tabular FFT features from history. It prefers
`scaled_history` when available and falls back to `raw_history`. The intended
path is to feed these features into tabular models through an explicit binder.

Typical use:

```text
MeanAbsScaling -> FourierFeatures -> BindFourierFeatures -> rf_tabular
MeanAbsScaling -> FourierFeatures -> BindFourierFeatures -> lightgbm_tabular
```

`lightgbm_tabular` loads LightGBM lazily, so the package can import without the
optional dependency. The Colab notebook installs `lightgbm` before running the
LightGBM comparisons.

### `kernel_rbf`

File: `graph_Time_series/token_blocks/kernel_rbf.py`

RBF KernelRidge model token. It reads the active `model_input` if one exists,
otherwise falls back to `scaled_history`. It accepts bundle kinds:

```text
sequence_flat
tabular
```

It fits leave-one-out predictions inside a single state and pushes those
predictions onto the residual stack.

## Default Grammar

The current built-in grammar is registered by:

```python
from graph_Time_series import Grammar
from graph_Time_series.token_blocks import register_default_tokens

grammar = register_default_tokens(Grammar())
```

The default token vocabulary is:

```text
ZNormalization
BindScaledHistory
BindAllSafeTabular
kernel_rbf
STOP
```

Example:

```python
from graph_Time_series import State, Grammar, apply_pipeline
from graph_Time_series.token_blocks import register_default_tokens

grammar = register_default_tokens(Grammar())
state = State(H, F)
state = apply_pipeline(
    "ZNormalization -> BindScaledHistory -> kernel_rbf -> STOP",
    grammar,
    state,
)

forecast = state.features["final_forecast"]
```

## Optional Lark Pipeline DSL

`pipeline_ast.py` adds a lightweight parser for linear token chains:

```python
from graph_Time_series import parse_pipeline, pipeline_to_sequence

ast = parse_pipeline("ZNormalization -> BindScaledHistory -> kernel_rbf -> STOP")
print(ast.names)
print(ast.to_source())
```

Install the optional parser dependency when needed:

```bash
pip install lark
```

The parser is deliberately non-invasive. It validates token names and graph
transitions, then delegates execution to the existing token instances.

Token keyword arguments are parsed into the AST, but not applied dynamically yet.
For now, configured variants should be registered as explicit token instances in
the grammar.

## Notebook Playground

The main working notebook is:

```text
examples/test_token_sequence_colab.ipynb
```

It currently includes:

- dataset loading through the companion exploratory repo;
- full state inspection and token-by-token diffs;
- notebook-local prototype tokens;
- artifact and bundle tables;
- leakage-safe holdout comparison;
- optional cross-validation;
- plots for manual and holdout forecasts.

Notebook-local prototype tokens include:

| Token | Status | Purpose |
| --- | --- | --- |
| `DataAugmentation` | notebook prototype | Jitter/smooth history for training experiments |
| `ContextWindow` | notebook prototype | Keep a fixed recent context window |
| `PeriodSelection` | notebook prototype | Pick a simple candidate period |
| `PeriodPhaseOneHot` | package token used in notebook | Encode phase inside selected period |
| `FourierFeatures` | package token used in notebook | Fixed-width FFT tabular features |
| `PeriodFold` | notebook prototype | Fold history into period phases |
| `ShapeLevel` | notebook prototype | Estimate a within-period shape vector |
| `shape_naive` | notebook prototype | Predict last level distributed through shape |
| `kernel_rbf_fast` | notebook prototype | Faster in-sample RBF diagnostic token |
| `level_kernel_rbf` | notebook prototype | RBF on level features, expanded through shape |
| `rf_tabular` | package token used in notebook | Random forest on tabular model input |
| `lightgbm_tabular` | package token used in notebook | LightGBM on tabular model input |

The notebook distinguishes manual in-sample diagnostics from leakage-safe
holdout scoring. Use the leakage-safe section for model comparison.

## State Evolution Example

For this sequence:

```text
ContextWindow -> ZNormalization -> BindScaledHistory -> kernel_rbf_fast
```

the state evolves as:

```text
State(H, F)
  raw_history registered as an artifact
  active_target_base = F
  current_target = F

ContextWindow
  original_history becomes the last context window
  features["raw_history"] is updated
  historical_features["context_history"] is added

ZNormalization
  historical_features["scaled_history"] is added
  active_target_base becomes normalized future
  current_target becomes normalized residual target
  inverse transform is registered

BindScaledHistory
  historical_features["model_input"] is set
  active InputBundle is set to "scaled_history"

kernel_rbf_fast
  reads model_input and current_target
  pushes one prediction
  current_target becomes the remaining residual
```

Package `kernel_rbf` follows the same state contract, with explicit leave-one-out
prediction inside the current state.

## Leakage-Safe Evaluation

For learned models, comparison should fit on training futures only and predict
separate holdout histories. The notebook's leakage-safe section does this by:

1. applying feature/binder tokens separately to train and query states;
2. fitting learned model wrappers on train features and train targets;
3. pushing predictions only into the query state;
4. ranking candidates by holdout metrics.

This avoids the older pattern where a model could accidentally fit on the same
future values used for evaluation.

## Adding New Tokens

New tokens should keep the broad-state plus explicit-binder pattern:

1. Feature tokens can create named artifacts.
2. Binder tokens decide which artifacts become `model_input`.
3. Model tokens consume declared bundle kinds.
4. Models call `state.push_prediction(...)` so residual chaining stays correct.
5. Transforms call `state.register_transform(...)` so final predictions decode
   correctly.

Minimal feature token sketch:

```python
from graph_Time_series.token import FeatureToken


class MyFeatureToken(FeatureToken):
    name = "MyFeature"
    reads = ("raw_history",)
    writes = ("my_feature",)

    def apply(self, state):
        value = ...  # shape must start with state.n_samples
        state.add_historical_feature("my_feature", value)
        self._log_execution(
            state,
            reads={"raw_history": state.features["raw_history"].shape},
            writes={"my_feature": value.shape},
        )
        return state
```

Minimal model token sketch:

```python
from graph_Time_series.token import ModelToken


class MyModelToken(ModelToken):
    name = "my_model"
    accepted_input_kinds = {"tabular"}

    def get_model(self):
        return {"model": "my_model"}

    def apply(self, state):
        X = state.historical_features["model_input"]
        Y = state.current_target
        pred = ...  # same shape as Y
        state.push_prediction(pred, self.name)
        self._log_execution(
            state,
            reads={"model_input": X.shape, "current_target": Y.shape},
            writes={"prediction_stack[-1]": pred.shape},
        )
        return state
```

## Open Extension Directions

These are intentionally left open-ended. They reflect directions explored in
discussion or the notebook, but are not fully promoted to package tokens yet.

| Direction | Possible token family |
| --- | --- |
| Context selection | `ContextWindow`, adaptive windows, multi-resolution windows |
| Period/seasonality features | `PeriodSelection`, `PeriodFold`, phase embeddings |
| Shape/level decomposition | `ShapeLevel`, level models, residual shape correction |
| Rich tabular models | linear/ridge, random forest, LightGBM, gradient boosting, XGBoost |
| Multi-channel sequence models | binders with `sequence_multi` or `tensor` bundle kinds |
| Residual ensembles | multiple model tokens chained through `current_target` |
| Calendar/known-future covariates | future feature artifacts and horizon-aware binders |
| Kernel variants | DTW kernels, shape kernels, level kernels, hybrid kernels |
| Search | grammar-guided candidate enumeration or MCTS over registered tokens |
| Configurable AST tokens | parsed keyword arguments mapped to configured token instances |

The most important rule for all of these is to keep token interfaces explicit:
create artifacts broadly, bind inputs deliberately, and let the state manage
transforms and residuals.

## Quick Smoke Test

```python
import numpy as np

from graph_Time_series import State, Grammar, apply_pipeline
from graph_Time_series.token_blocks import register_default_tokens

H = np.random.randn(16, 48).astype("float32")
F = np.random.randn(16, 12).astype("float32")

grammar = register_default_tokens(Grammar())
state = State(H, F)
state = apply_pipeline(
    "ZNormalization -> BindScaledHistory -> kernel_rbf -> STOP",
    grammar,
    state,
)

print(state.token_sequence)
print(state.features["final_forecast"].shape)
print(state.describe_artifacts())
print(state.describe_input_bundles())
```
