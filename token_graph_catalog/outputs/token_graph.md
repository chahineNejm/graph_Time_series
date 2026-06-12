# Token Graph

Project: `graph_Time_series`

This Markdown view is aligned with the live package registries as of the
signal-board, binder-free architecture. It reflects:

```python
register_flair_gb_swap(
    register_versatile_tokens(
        register_flair_tokens(
            register_default_tokens(Grammar())
        )
    )
)
```

Current live grammar:

```text
Grammar(20 tokens, 58 edges)
```

Note: graph edges are syntactic possibilities. `Token.can_apply(state)` still
enforces usage caps, dependency checks, semantic ports, and model-specific
conditions. For example, `FourierFeatures` has a self-edge in the grammar, but
its `max_uses=1` prevents repeated application in a real state.

## Core Graph

```mermaid
flowchart LR
  n_START["START<br/>control"]:::control
  n_STOP["STOP<br/>control"]:::control
  n_FlairPreprocess["FlairPreprocess<br/>cleaning"]:::cleaning
  n_MeanAbsScaling["MeanAbsScaling<br/>transform"]:::transform
  n_ZNormalization["ZNormalization<br/>transform"]:::transform
  n_DayOfWeekFeature["DayOfWeekFeature<br/>feature"]:::feature
  n_FourierFeatures["FourierFeatures<br/>feature"]:::feature
  n_LevelBoxCoxCenter["LevelBoxCoxCenter<br/>feature"]:::feature
  n_LevelShrinkage["LevelShrinkage<br/>feature"]:::feature
  n_PeriodFold["PeriodFold<br/>feature"]:::feature
  n_PeriodPhaseOneHot["PeriodPhaseOneHot<br/>feature"]:::feature
  n_PeriodSelection["PeriodSelection<br/>feature"]:::feature
  n_SecondaryLevelSeasonality["SecondaryLevelSeasonality<br/>feature"]:::feature
  n_ShapeLevel["ShapeLevel<br/>feature"]:::feature
  n_FlairRidgeLevel["FlairRidgeLevel<br/>model"]:::model
  n_FlairSamplePaths["FlairSamplePaths<br/>feature"]:::feature
  n_gb_level_forecast["gb_level_forecast<br/>model"]:::model
  n_kernel_rbf["kernel_rbf<br/>model"]:::model
  n_lightgbm_tabular["lightgbm_tabular<br/>model"]:::model
  n_rf_tabular["rf_tabular<br/>model"]:::model
  n_versatile_gb["versatile_gb<br/>model"]:::model
  n_versatile_rf["versatile_rf<br/>model"]:::model

  n_START --> n_DayOfWeekFeature
  n_START --> n_FlairPreprocess
  n_START --> n_FourierFeatures
  n_START --> n_MeanAbsScaling
  n_START --> n_ZNormalization
  n_START --> n_versatile_gb
  n_START --> n_versatile_rf

  n_ZNormalization --> n_DayOfWeekFeature
  n_ZNormalization --> n_FourierFeatures
  n_ZNormalization --> n_kernel_rbf
  n_ZNormalization --> n_lightgbm_tabular
  n_ZNormalization --> n_rf_tabular
  n_ZNormalization --> n_versatile_gb
  n_ZNormalization --> n_versatile_rf

  n_MeanAbsScaling --> n_DayOfWeekFeature
  n_MeanAbsScaling --> n_FourierFeatures
  n_MeanAbsScaling --> n_kernel_rbf
  n_MeanAbsScaling --> n_lightgbm_tabular
  n_MeanAbsScaling --> n_rf_tabular
  n_MeanAbsScaling --> n_versatile_gb
  n_MeanAbsScaling --> n_versatile_rf

  n_DayOfWeekFeature --> n_DayOfWeekFeature
  n_DayOfWeekFeature --> n_FourierFeatures
  n_DayOfWeekFeature --> n_versatile_gb
  n_DayOfWeekFeature --> n_versatile_rf

  n_FourierFeatures --> n_DayOfWeekFeature
  n_FourierFeatures --> n_FourierFeatures
  n_FourierFeatures --> n_kernel_rbf
  n_FourierFeatures --> n_lightgbm_tabular
  n_FourierFeatures --> n_rf_tabular
  n_FourierFeatures --> n_versatile_gb
  n_FourierFeatures --> n_versatile_rf

  n_kernel_rbf --> n_kernel_rbf
  n_kernel_rbf --> n_STOP
  n_rf_tabular --> n_STOP
  n_lightgbm_tabular --> n_STOP

  n_versatile_rf --> n_versatile_rf
  n_versatile_rf --> n_versatile_gb
  n_versatile_rf --> n_STOP
  n_versatile_gb --> n_versatile_rf
  n_versatile_gb --> n_versatile_gb
  n_versatile_gb --> n_STOP

  n_FlairPreprocess --> n_PeriodSelection
  n_PeriodSelection --> n_PeriodPhaseOneHot
  n_PeriodSelection --> n_PeriodFold
  n_PeriodPhaseOneHot --> n_PeriodFold
  n_PeriodFold --> n_LevelShrinkage
  n_PeriodFold --> n_ShapeLevel
  n_LevelShrinkage --> n_ShapeLevel
  n_LevelShrinkage --> n_gb_level_forecast
  n_ShapeLevel --> n_SecondaryLevelSeasonality
  n_ShapeLevel --> n_gb_level_forecast
  n_SecondaryLevelSeasonality --> n_LevelBoxCoxCenter
  n_LevelBoxCoxCenter --> n_FlairRidgeLevel
  n_FlairRidgeLevel --> n_FlairSamplePaths
  n_FlairRidgeLevel --> n_STOP
  n_FlairSamplePaths --> n_STOP
  n_gb_level_forecast --> n_STOP

  classDef control fill:#ffe2e2,stroke:#b91c1c,stroke-width:1px,color:#111827
  classDef cleaning fill:#dff0ff,stroke:#2563eb,stroke-width:1px,color:#111827
  classDef transform fill:#fff1c2,stroke:#b45309,stroke-width:1px,color:#111827
  classDef feature fill:#d8f7ef,stroke:#047857,stroke-width:1px,color:#111827
  classDef model fill:#ffe6cf,stroke:#ea580c,stroke-width:1px,color:#111827
```

## Token Matrix

| Token | Class | Parents | Next |
| --- | --- | --- | --- |
| `START` | control |  | `DayOfWeekFeature`, `FlairPreprocess`, `FourierFeatures`, `MeanAbsScaling`, `ZNormalization`, `versatile_gb`, `versatile_rf` |
| `STOP` | control | `FlairRidgeLevel`, `FlairSamplePaths`, `gb_level_forecast`, `kernel_rbf`, `lightgbm_tabular`, `rf_tabular`, `versatile_gb`, `versatile_rf` |  |
| `DayOfWeekFeature` | feature | `DayOfWeekFeature`, `FourierFeatures`, `MeanAbsScaling`, `START`, `ZNormalization` | `DayOfWeekFeature`, `FourierFeatures`, `versatile_gb`, `versatile_rf` |
| `FlairPreprocess` | cleaning | `START` | `PeriodSelection` |
| `FlairRidgeLevel` | model | `LevelBoxCoxCenter` | `FlairSamplePaths`, `STOP` |
| `FlairSamplePaths` | feature | `FlairRidgeLevel` | `STOP` |
| `FourierFeatures` | feature | `DayOfWeekFeature`, `FourierFeatures`, `MeanAbsScaling`, `START`, `ZNormalization` | `DayOfWeekFeature`, `FourierFeatures`, `kernel_rbf`, `lightgbm_tabular`, `rf_tabular`, `versatile_gb`, `versatile_rf` |
| `LevelBoxCoxCenter` | feature | `SecondaryLevelSeasonality` | `FlairRidgeLevel` |
| `LevelShrinkage` | feature | `PeriodFold` | `ShapeLevel`, `gb_level_forecast` |
| `MeanAbsScaling` | transform | `START` | `DayOfWeekFeature`, `FourierFeatures`, `kernel_rbf`, `lightgbm_tabular`, `rf_tabular`, `versatile_gb`, `versatile_rf` |
| `PeriodFold` | feature | `PeriodPhaseOneHot`, `PeriodSelection` | `LevelShrinkage`, `ShapeLevel` |
| `PeriodPhaseOneHot` | feature | `PeriodSelection` | `PeriodFold` |
| `PeriodSelection` | feature | `FlairPreprocess` | `PeriodFold`, `PeriodPhaseOneHot` |
| `SecondaryLevelSeasonality` | feature | `ShapeLevel` | `LevelBoxCoxCenter` |
| `ShapeLevel` | feature | `LevelShrinkage`, `PeriodFold` | `SecondaryLevelSeasonality`, `gb_level_forecast` |
| `ZNormalization` | transform | `START` | `DayOfWeekFeature`, `FourierFeatures`, `kernel_rbf`, `lightgbm_tabular`, `rf_tabular`, `versatile_gb`, `versatile_rf` |
| `gb_level_forecast` | model | `LevelShrinkage`, `ShapeLevel` | `STOP` |
| `kernel_rbf` | model | `FourierFeatures`, `MeanAbsScaling`, `ZNormalization`, `kernel_rbf` | `STOP`, `kernel_rbf` |
| `lightgbm_tabular` | model | `FourierFeatures`, `MeanAbsScaling`, `ZNormalization` | `STOP` |
| `rf_tabular` | model | `FourierFeatures`, `MeanAbsScaling`, `ZNormalization` | `STOP` |
| `versatile_gb` | model | `DayOfWeekFeature`, `FourierFeatures`, `MeanAbsScaling`, `START`, `ZNormalization`, `versatile_gb`, `versatile_rf` | `STOP`, `versatile_gb`, `versatile_rf` |
| `versatile_rf` | model | `DayOfWeekFeature`, `FourierFeatures`, `MeanAbsScaling`, `START`, `ZNormalization`, `versatile_gb`, `versatile_rf` | `STOP`, `versatile_gb`, `versatile_rf` |

## Legacy Catalog Warning

This file is currently the Markdown view aligned with the live package grammar.
The adjacent `token_catalog.json` may still contain older manual entries until
the catalog is refreshed.
