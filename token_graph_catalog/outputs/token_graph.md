# Token Graph

Project: `graph_Time_series`

Generated from the live package registries, not the manual catalog JSON.

Registry stack:

```python
register_seasonal_tokens(
    register_flair_gb_swap(
        register_versatile_tokens(
            register_flair_tokens(
                register_default_tokens(Grammar())
            )
        )
    )
)
```

Current live grammar:

```text
Grammar(20 tokens, 57 edges)
```

## Graph

```mermaid
flowchart LR
  n_START["START<br/>START"]:::START
  n_FlairPreprocess["FlairPreprocess<br/>cleaning"]:::cleaning
  n_STOP["STOP<br/>control"]:::control
  n_FlairSamplePaths["FlairSamplePaths<br/>feature"]:::feature
  n_FourierFeatures["FourierFeatures<br/>feature"]:::feature
  n_LevelBoxCoxCenter["LevelBoxCoxCenter<br/>feature"]:::feature
  n_PeriodDetect["PeriodDetect<br/>feature"]:::feature
  n_PeriodFold["PeriodFold<br/>feature"]:::feature
  n_PeriodPhaseOneHot["PeriodPhaseOneHot<br/>feature"]:::feature
  n_SeasonalFeatures["SeasonalFeatures<br/>feature"]:::feature
  n_SecondaryLevelSeasonality["SecondaryLevelSeasonality<br/>feature"]:::feature
  n_ShapeLevel["ShapeLevel<br/>feature"]:::feature
  n_FlairRidgeLevel["FlairRidgeLevel<br/>model"]:::model
  n_gb_level_forecast["gb_level_forecast<br/>model"]:::model
  n_kernel_rbf["kernel_rbf<br/>model"]:::model
  n_level_shape_ridge["level_shape_ridge<br/>model"]:::model
  n_lightgbm_tabular["lightgbm_tabular<br/>model"]:::model
  n_rf_tabular["rf_tabular<br/>model"]:::model
  n_step_regression["step_regression<br/>model"]:::model
  n_versatile_gb["versatile_gb<br/>model"]:::model
  n_MeanAbsScaling["MeanAbsScaling<br/>transform"]:::transform
  n_ZNormalization["ZNormalization<br/>transform"]:::transform
  n_FlairPreprocess --> n_PeriodDetect
  n_FlairRidgeLevel --> n_FlairSamplePaths
  n_FlairRidgeLevel --> n_STOP
  n_FlairSamplePaths --> n_STOP
  n_FourierFeatures --> n_FourierFeatures
  n_FourierFeatures --> n_kernel_rbf
  n_FourierFeatures --> n_lightgbm_tabular
  n_FourierFeatures --> n_rf_tabular
  n_FourierFeatures --> n_step_regression
  n_FourierFeatures --> n_versatile_gb
  n_LevelBoxCoxCenter --> n_FlairRidgeLevel
  n_MeanAbsScaling --> n_FourierFeatures
  n_MeanAbsScaling --> n_PeriodDetect
  n_MeanAbsScaling --> n_kernel_rbf
  n_MeanAbsScaling --> n_lightgbm_tabular
  n_MeanAbsScaling --> n_rf_tabular
  n_MeanAbsScaling --> n_step_regression
  n_MeanAbsScaling --> n_versatile_gb
  n_PeriodDetect --> n_PeriodFold
  n_PeriodDetect --> n_PeriodPhaseOneHot
  n_PeriodDetect --> n_SeasonalFeatures
  n_PeriodDetect --> n_step_regression
  n_PeriodFold --> n_ShapeLevel
  n_PeriodPhaseOneHot --> n_PeriodFold
  n_START --> n_FlairPreprocess
  n_START --> n_FourierFeatures
  n_START --> n_MeanAbsScaling
  n_START --> n_PeriodDetect
  n_START --> n_ZNormalization
  n_START --> n_versatile_gb
  n_SeasonalFeatures --> n_kernel_rbf
  n_SeasonalFeatures --> n_lightgbm_tabular
  n_SeasonalFeatures --> n_rf_tabular
  n_SeasonalFeatures --> n_step_regression
  n_SeasonalFeatures --> n_versatile_gb
  n_SecondaryLevelSeasonality --> n_LevelBoxCoxCenter
  n_SecondaryLevelSeasonality --> n_level_shape_ridge
  n_ShapeLevel --> n_SecondaryLevelSeasonality
  n_ShapeLevel --> n_gb_level_forecast
  n_ShapeLevel --> n_level_shape_ridge
  n_ZNormalization --> n_FourierFeatures
  n_ZNormalization --> n_PeriodDetect
  n_ZNormalization --> n_kernel_rbf
  n_ZNormalization --> n_lightgbm_tabular
  n_ZNormalization --> n_rf_tabular
  n_ZNormalization --> n_step_regression
  n_ZNormalization --> n_versatile_gb
  n_gb_level_forecast --> n_STOP
  n_kernel_rbf --> n_STOP
  n_kernel_rbf --> n_kernel_rbf
  n_level_shape_ridge --> n_STOP
  n_level_shape_ridge --> n_kernel_rbf
  n_lightgbm_tabular --> n_STOP
  n_rf_tabular --> n_STOP
  n_step_regression --> n_STOP
  n_versatile_gb --> n_STOP
  n_versatile_gb --> n_versatile_gb
  classDef control fill:#ffe2e2,stroke:#b91c1c,stroke-width:1px,color:#111827
  classDef START fill:#dcfce7,stroke:#166534,stroke-width:1px,color:#111827
  classDef cleaning fill:#dff0ff,stroke:#2563eb,stroke-width:1px,color:#111827
  classDef transform fill:#fff1c2,stroke:#b45309,stroke-width:1px,color:#111827
  classDef feature fill:#d8f7ef,stroke:#047857,stroke-width:1px,color:#111827
  classDef model fill:#ffe6cf,stroke:#ea580c,stroke-width:1px,color:#111827
```

## Token Matrix

| Token | Class | Parents | Next |
| --- | --- | --- | --- |
| `START` | START |  | `FlairPreprocess`, `FourierFeatures`, `MeanAbsScaling`, `PeriodDetect`, `ZNormalization`, `versatile_gb` |
| `FlairPreprocess` | cleaning | `START` | `PeriodDetect` |
| `STOP` | control | `FlairRidgeLevel`, `FlairSamplePaths`, `gb_level_forecast`, `kernel_rbf`, `level_shape_ridge`, `lightgbm_tabular`, `rf_tabular`, `step_regression`, `versatile_gb` |  |
| `FlairSamplePaths` | feature | `FlairRidgeLevel` | `STOP` |
| `FourierFeatures` | feature | `FourierFeatures`, `MeanAbsScaling`, `START`, `ZNormalization` | `FourierFeatures`, `kernel_rbf`, `lightgbm_tabular`, `rf_tabular`, `step_regression`, `versatile_gb` |
| `LevelBoxCoxCenter` | feature | `SecondaryLevelSeasonality` | `FlairRidgeLevel` |
| `PeriodDetect` | feature | `FlairPreprocess`, `MeanAbsScaling`, `START`, `ZNormalization` | `PeriodFold`, `PeriodPhaseOneHot`, `SeasonalFeatures`, `step_regression` |
| `PeriodFold` | feature | `PeriodDetect`, `PeriodPhaseOneHot` | `ShapeLevel` |
| `PeriodPhaseOneHot` | feature | `PeriodDetect` | `PeriodFold` |
| `SeasonalFeatures` | feature | `PeriodDetect` | `kernel_rbf`, `lightgbm_tabular`, `rf_tabular`, `step_regression`, `versatile_gb` |
| `SecondaryLevelSeasonality` | feature | `ShapeLevel` | `LevelBoxCoxCenter`, `level_shape_ridge` |
| `ShapeLevel` | feature | `PeriodFold` | `SecondaryLevelSeasonality`, `gb_level_forecast`, `level_shape_ridge` |
| `FlairRidgeLevel` | model | `LevelBoxCoxCenter` | `FlairSamplePaths`, `STOP` |
| `gb_level_forecast` | model | `ShapeLevel` | `STOP` |
| `kernel_rbf` | model | `FourierFeatures`, `MeanAbsScaling`, `SeasonalFeatures`, `ZNormalization`, `kernel_rbf`, `level_shape_ridge` | `STOP`, `kernel_rbf` |
| `level_shape_ridge` | model | `SecondaryLevelSeasonality`, `ShapeLevel` | `STOP`, `kernel_rbf` |
| `lightgbm_tabular` | model | `FourierFeatures`, `MeanAbsScaling`, `SeasonalFeatures`, `ZNormalization` | `STOP` |
| `rf_tabular` | model | `FourierFeatures`, `MeanAbsScaling`, `SeasonalFeatures`, `ZNormalization` | `STOP` |
| `step_regression` | model | `FourierFeatures`, `MeanAbsScaling`, `PeriodDetect`, `SeasonalFeatures`, `ZNormalization` | `STOP` |
| `versatile_gb` | model | `FourierFeatures`, `MeanAbsScaling`, `START`, `SeasonalFeatures`, `ZNormalization`, `versatile_gb` | `STOP`, `versatile_gb` |
| `MeanAbsScaling` | transform | `START` | `FourierFeatures`, `PeriodDetect`, `kernel_rbf`, `lightgbm_tabular`, `rf_tabular`, `step_regression`, `versatile_gb` |
| `ZNormalization` | transform | `START` | `FourierFeatures`, `PeriodDetect`, `kernel_rbf`, `lightgbm_tabular`, `rf_tabular`, `step_regression`, `versatile_gb` |
