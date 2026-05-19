# Graph Time Series — CLP Framework for Forecasting Pipeline Discovery

A framework for **automated time-series forecasting pipeline construction** using Monte Carlo Tree Search (MCTS). Based on the Computational Language Processing (CLP) paradigm: define small computational tokens, wire them into a grammar graph, then let MCTS find the best compositions.

## Architecture

```
State ──► Token ──► Token ──► ... ──► STOP
  │         │         │                 │
  │     (transforms   (transforms   (evaluates,
  │      features)     residuals)    sets MASE)
  │
  └── features dict, prediction_stack, current_target (auto-updated residual)
```

The framework has four core abstractions:

| Concept | File | Role |
|---------|------|------|
| **State** | `state.py` | RL state: holds history, future, features, prediction stack, transformation log |
| **Token** | `token.py` | Abstract base class — a single computational step |
| **Grammar** | `grammar.py` | Directed graph of valid token transitions |
| **MCTS** | `mcts.py` | AlphaZero-style tree search with PUCT over the grammar |

## Token classes

Every token belongs to a class that determines where it fits in a pipeline:

| Class | Purpose | Reads | Writes |
|-------|---------|-------|--------|
| `cleaning` | Preprocess raw history | `raw_history` | `cleaned` |
| `feature` | Extract features for models | `cleaned` | `model_input` |
| `encoder` | Transform to a different domain | `cleaned` | `model_input` + domain state |
| `model` | Fit and predict (LOO) | `model_input` | pushes to `prediction_stack` |
| `decoder` | Inverse-transform predictions | domain state | modifies `prediction_stack` |
| `control` | Sequence control (STOP) | `prediction_stack` | `final_forecast` + `mase` |

A valid pipeline always follows this pattern:

```
START → cleaning → feature/encoder → model [→ model → ...] → [decoder →] STOP
```

## How to create a new token

### 1. Write the token class

Create a new class that inherits from `Token`. You must define five attributes and implement `apply()`:

```python
# tokens/my_new_tokens.py
import numpy as np
from ..token import Token, _shapes


class CleanZScore(Token):
    name = "zscore"                      # unique identifier
    token_class = "cleaning"             # determines grammar position
    reads = ["raw_history"]              # required keys in state.features
    writes = ["cleaned"]                 # keys this token creates
    description = "Per-sample z-score normalisation"

    def apply(self, state):
        X = state.features["raw_history"]
        mu = X.mean(axis=1, keepdims=True)
        sigma = X.std(axis=1, keepdims=True) + 1e-8
        state.features["cleaned"] = (X - mu) / sigma

        # Always log what happened
        state.log_step(self.name,
                       _shapes(state, self.reads),
                       {k: state.features[k].shape for k in self.writes})
        state.token_sequence.append(self.name)
        return state
```

### 2. Register it in the grammar

In `tokens/__init__.py`, import your token and add it to `register_all()`:

```python
from .my_new_tokens import CleanZScore

def register_all(grammar):
    # ... existing registrations ...

    # New cleaning token: follows START (like all cleaning tokens)
    grammar.register(CleanZScore(), follows=["START"])
```

That's it. MCTS will now explore pipelines that use your token.

## Token templates by class

### Cleaning token

Reads `raw_history`, writes `cleaned`. Follows `START`.

```python
class CleanExample(Token):
    name = "my_cleaner"
    token_class = "cleaning"
    reads = ["raw_history"]
    writes = ["cleaned"]
    description = "What this cleaner does"

    def apply(self, state):
        X = state.features["raw_history"]
        # ... transform X ...
        state.features["cleaned"] = result
        state.log_step(self.name, _shapes(state, self.reads),
                       {"cleaned": result.shape})
        state.token_sequence.append(self.name)
        return state
```

Register: `grammar.register(CleanExample(), follows=["START"])`

### Feature token

Reads `cleaned`, writes `model_input`. Follows cleaning tokens.

```python
class FeatExample(Token):
    name = "my_features"
    token_class = "feature"
    reads = ["cleaned"]
    writes = ["model_input"]
    description = "What features this extracts"

    def apply(self, state):
        X = state.features["cleaned"]
        # ... extract features ...
        state.features["model_input"] = features
        state.log_step(self.name, _shapes(state, self.reads),
                       {"model_input": features.shape})
        state.token_sequence.append(self.name)
        return state
```

Register: `grammar.register(FeatExample(), follows=["identity", "detrend", "moving_avg"])`

### Model token

Reads `model_input`, uses `state.current_target` as the Y to fit. Must call `state.push_prediction()` which auto-updates the residual for chaining.

```python
class ModelExample(Token):
    name = "my_model"
    token_class = "model"
    reads = ["model_input"]
    writes = []
    description = "What this model does"

    def apply(self, state):
        X = state.features["model_input"].astype(np.float32)
        Y = state.current_target.astype(np.float32)  # residual if chained
        n = X.shape[0]
        Y_loo = np.zeros_like(Y)

        # Leave-one-out evaluation
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            # ... fit on X[mask], Y[mask] ...
            # ... predict X[i:i+1] ...
            Y_loo[i] = prediction

        # This auto-updates state.current_target for the next model
        state.push_prediction(Y_loo, self.name)
        state.log_step(self.name,
                       {"model_input": X.shape, "current_target": Y.shape},
                       {"prediction_stack[-1]": Y_loo.shape})
        state.token_sequence.append(self.name)
        return state
```

Register (with residual chaining):
```python
model_follows = ["feat_raw", "fft_encode", "kernel_rbf", "random_forest", "xgboost", "my_model"]
model_leads = ["kernel_rbf", "random_forest", "xgboost", "my_model", "fft_decode", "STOP"]
grammar.register(ModelExample(), follows=model_follows, leads_to=model_leads)
```

### Encoder / Decoder pair

Encoders transform to a different domain (e.g. FFT). Decoders transform back. The encoder stores state that the decoder needs.

```python
class EncodeExample(Token):
    name = "my_encode"
    token_class = "encoder"
    reads = ["cleaned"]
    writes = ["model_input", "encode_state"]  # store whatever the decoder needs
    description = "Encode to some domain"

    def apply(self, state):
        X = state.features["cleaned"]
        # ... transform ...
        state.features["model_input"] = transformed
        state.features["encode_state"] = side_info  # for the decoder
        state.log_step(self.name, _shapes(state, self.reads),
                       {k: state.features[k].shape for k in self.writes})
        state.token_sequence.append(self.name)
        return state

class DecodeExample(Token):
    name = "my_decode"
    token_class = "decoder"
    reads = ["encode_state"]
    writes = []
    description = "Decode back from that domain"

    def apply(self, state):
        side_info = state.features["encode_state"]
        pred = state.last_prediction()
        # ... inverse transform pred using side_info ...
        state.prediction_stack[-1] = reconstructed
        state.log_step(self.name, _shapes(state, self.reads), {})
        state.token_sequence.append(self.name)
        return state
```

## Grammar wiring

The grammar is a directed graph. `register()` takes:
- `follows`: list of token names that can appear **before** this token
- `leads_to`: list of token names that can appear **after** this token

```python
grammar = Grammar()

# A follows B means: after B runs, A becomes a valid action
grammar.register(my_token, follows=["identity", "detrend"], leads_to=["STOP"])
```

The grammar enforces two constraints simultaneously:
1. **Graph edges** — the token must be reachable from the current position
2. **Feature availability** — `token.reads` must all exist in `state.features`

STOP is a special control token: it only becomes valid when at least one model has been applied (`state.n_models_applied > 0`).

## Residual chaining

When a model calls `state.push_prediction(Y_loo, name)`, the State automatically updates `current_target`:

```
current_target = original_future - sum(prediction_stack)
```

So the second model in a chain fits the **residual** of the first, the third fits the residual of the first two, etc. The final forecast is the sum of all predictions.

## MCTS search

```python
from graph_Time_series import State, Grammar, mcts_search
from graph_Time_series.tokens import register_all

grammar = Grammar()
register_all(grammar)

state = State(history_array, future_array)  # both (n_samples, length)
results = mcts_search(grammar, state, n_iterations=40, puct_c=1.5)

print(results["best_chain"])   # e.g. "identity -> feat_raw -> kernel_rbf -> STOP"
print(results["best_mase"])    # e.g. 2.34
```

The search uses the PUCT formula (AlphaZero-style) with MI-based priors from `heuristics.py`:

```
score(a) = Q(a) + c * prior(a) * sqrt(N_parent) / (1 + N(a))
```

## Existing tokens

| Name | Class | Description |
|------|-------|-------------|
| `identity` | cleaning | Pass-through |
| `detrend` | cleaning | Remove per-sample linear trend |
| `moving_avg` | cleaning | Centred moving average (window=5) |
| `feat_raw` | feature | Use cleaned histories directly |
| `fft_encode` | encoder | FFT magnitudes as features, store phase |
| `fft_decode` | decoder | Inverse FFT (stub — to implement) |
| `kernel_rbf` | model | RBF kernel regression, analytic LOO |
| `random_forest` | model | Random forest, explicit LOO |
| `xgboost` | model | XGBoost, explicit LOO |
| `STOP` | control | Compute final forecast and MASE |

## Quick start

```bash
pip install numpy scikit-learn xgboost networkx matplotlib tqdm datasets
```

```python
import numpy as np
from graph_Time_series import State, Grammar, mcts_search, plot_grammar
from graph_Time_series.tokens import register_all

# Build grammar
grammar = Grammar()
register_all(grammar)
plot_grammar(grammar)

# Load your data as (n_samples, history_length) and (n_samples, horizon)
H = np.random.randn(25, 48).astype(np.float32)
F = np.random.randn(25, 12).astype(np.float32)

# Run search
state = State(H, F)
results = mcts_search(grammar, state, n_iterations=40)

# Inspect
print(results["best_chain"], results["best_mase"])
```

## File structure

```
graph_Time_series/
├── __init__.py          # package exports
├── state.py             # State (RL state)
├── token.py             # Token ABC
├── grammar.py           # Grammar graph + visualisation
├── heuristics.py        # MI-based PUCT priors
├── mcts.py              # MCTS search with tqdm progress bars
├── README.md            # this file
├── run_colab.ipynb      # ready-to-run Colab notebook
└── tokens/
    ├── __init__.py      # register_all(grammar)
    ├── cleaning.py      # identity, detrend, moving_avg
    ├── features.py      # feat_raw, fft_encode, fft_decode (stub)
    └── models.py        # kernel_rbf, random_forest, xgboost, STOP
```
