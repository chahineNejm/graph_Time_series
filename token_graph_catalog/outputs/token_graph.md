# Token Graph

Project: `graph_Time_series`

Generated from `token_catalog.json`.

## Core Graph

Core view hides generic binders, aliases, and broad experimental adapters.

```mermaid
flowchart LR
  n_START["START<br/>synthetic / control"]:::control
  n_FlairPreprocess["FlairPreprocess<br/>package / cleaning"]:::cleaning
  n_ForwardFill["ForwardFill<br/>package / cleaning"]:::cleaning
  n_LinearFill["LinearFill<br/>package / cleaning"]:::cleaning
  n_MeanAbsScaling["MeanAbsScaling<br/>package / transform"]:::transform
  n_ZNormalization["ZNormalization<br/>package / transform"]:::transform
  n_FlairSamplePaths["FlairSamplePaths<br/>package / feature"]:::feature
  n_FourierFeatures["FourierFeatures<br/>package / feature"]:::feature
  n_PeriodDetect["PeriodDetect<br/>package / feature"]:::feature
  n_PeriodDetectBIC["PeriodDetectBIC<br/>package / feature"]:::feature
  n_PeriodDetectSpectral["PeriodDetectSpectral<br/>package / feature"]:::feature
  n_PeriodPhaseOneHot["PeriodPhaseOneHot<br/>package / feature"]:::feature
  n_SeasonalFeatures["SeasonalFeatures<br/>package / feature"]:::feature
  n_SeasonalFold["SeasonalFold<br/>package / feature"]:::feature
  n_FlairRidgeLevel["FlairRidgeLevel<br/>package / model"]:::model
  n_gb_level_forecast["gb_level_forecast<br/>package / model"]:::model
  n_kernel_rbf["kernel_rbf<br/>package / model"]:::model
  n_level_shape_ridge["level_shape_ridge<br/>package / model"]:::model
  n_lightgbm_tabular["lightgbm_tabular<br/>package / model"]:::model
  n_parrot["parrot<br/>package / model"]:::model
  n_parrot_dataset["parrot_dataset<br/>package / model"]:::model
  n_rf_tabular["rf_tabular<br/>package / model"]:::model
  n_step_regression["step_regression<br/>package / model"]:::model
  n_versatile_gb["versatile_gb<br/>package / model"]:::model
  n_STOP["STOP<br/>synthetic / terminal"]:::terminal
  n_FlairPreprocess --> n_PeriodDetect
  n_FlairPreprocess --> n_PeriodDetectBIC
  n_FlairPreprocess --> n_PeriodDetectSpectral
  n_FlairRidgeLevel --> n_FlairSamplePaths
  n_FlairRidgeLevel --> n_STOP
  n_FlairRidgeLevel --> n_parrot
  n_FlairSamplePaths --> n_STOP
  n_ForwardFill --> n_FourierFeatures
  n_ForwardFill --> n_MeanAbsScaling
  n_ForwardFill --> n_PeriodDetect
  n_ForwardFill --> n_PeriodDetectBIC
  n_ForwardFill --> n_PeriodDetectSpectral
  n_ForwardFill --> n_ZNormalization
  n_ForwardFill --> n_kernel_rbf
  n_ForwardFill --> n_lightgbm_tabular
  n_ForwardFill --> n_parrot
  n_ForwardFill --> n_parrot_dataset
  n_ForwardFill --> n_rf_tabular
  n_ForwardFill --> n_versatile_gb
  n_FourierFeatures --> n_FourierFeatures
  n_FourierFeatures --> n_kernel_rbf
  n_FourierFeatures --> n_lightgbm_tabular
  n_FourierFeatures --> n_parrot
  n_FourierFeatures --> n_parrot_dataset
  n_FourierFeatures --> n_rf_tabular
  n_FourierFeatures --> n_step_regression
  n_FourierFeatures --> n_versatile_gb
  n_LinearFill --> n_FourierFeatures
  n_LinearFill --> n_MeanAbsScaling
  n_LinearFill --> n_PeriodDetect
  n_LinearFill --> n_PeriodDetectBIC
  n_LinearFill --> n_PeriodDetectSpectral
  n_LinearFill --> n_ZNormalization
  n_LinearFill --> n_kernel_rbf
  n_LinearFill --> n_lightgbm_tabular
  n_LinearFill --> n_parrot
  n_LinearFill --> n_parrot_dataset
  n_LinearFill --> n_rf_tabular
  n_LinearFill --> n_versatile_gb
  n_MeanAbsScaling --> n_FourierFeatures
  n_MeanAbsScaling --> n_PeriodDetect
  n_MeanAbsScaling --> n_PeriodDetectBIC
  n_MeanAbsScaling --> n_PeriodDetectSpectral
  n_MeanAbsScaling --> n_kernel_rbf
  n_MeanAbsScaling --> n_lightgbm_tabular
  n_MeanAbsScaling --> n_parrot
  n_MeanAbsScaling --> n_parrot_dataset
  n_MeanAbsScaling --> n_rf_tabular
  n_MeanAbsScaling --> n_step_regression
  n_MeanAbsScaling --> n_versatile_gb
  n_PeriodDetect --> n_PeriodPhaseOneHot
  n_PeriodDetect --> n_SeasonalFeatures
  n_PeriodDetect --> n_SeasonalFold
  n_PeriodDetect --> n_step_regression
  n_PeriodDetectBIC --> n_PeriodPhaseOneHot
  n_PeriodDetectBIC --> n_SeasonalFeatures
  n_PeriodDetectBIC --> n_SeasonalFold
  n_PeriodDetectBIC --> n_step_regression
  n_PeriodDetectSpectral --> n_PeriodPhaseOneHot
  n_PeriodDetectSpectral --> n_SeasonalFeatures
  n_PeriodDetectSpectral --> n_SeasonalFold
  n_PeriodDetectSpectral --> n_step_regression
  n_PeriodPhaseOneHot --> n_SeasonalFold
  n_START --> n_FlairPreprocess
  n_START --> n_ForwardFill
  n_START --> n_FourierFeatures
  n_START --> n_LinearFill
  n_START --> n_MeanAbsScaling
  n_START --> n_PeriodDetect
  n_START --> n_PeriodDetectBIC
  n_START --> n_PeriodDetectSpectral
  n_START --> n_ZNormalization
  n_START --> n_versatile_gb
  n_SeasonalFeatures --> n_kernel_rbf
  n_SeasonalFeatures --> n_lightgbm_tabular
  n_SeasonalFeatures --> n_parrot
  n_SeasonalFeatures --> n_rf_tabular
  n_SeasonalFeatures --> n_step_regression
  n_SeasonalFeatures --> n_versatile_gb
  n_SeasonalFold --> n_FlairRidgeLevel
  n_SeasonalFold --> n_SeasonalFold
  n_SeasonalFold --> n_gb_level_forecast
  n_SeasonalFold --> n_level_shape_ridge
  n_ZNormalization --> n_FourierFeatures
  n_ZNormalization --> n_PeriodDetect
  n_ZNormalization --> n_PeriodDetectBIC
  n_ZNormalization --> n_PeriodDetectSpectral
  n_ZNormalization --> n_kernel_rbf
  n_ZNormalization --> n_lightgbm_tabular
  n_ZNormalization --> n_parrot
  n_ZNormalization --> n_parrot_dataset
  n_ZNormalization --> n_rf_tabular
  n_ZNormalization --> n_step_regression
  n_ZNormalization --> n_versatile_gb
  n_gb_level_forecast --> n_STOP
  n_gb_level_forecast --> n_parrot
  n_kernel_rbf --> n_STOP
  n_kernel_rbf --> n_kernel_rbf
  n_kernel_rbf --> n_parrot
  n_kernel_rbf --> n_parrot_dataset
  n_level_shape_ridge --> n_STOP
  n_level_shape_ridge --> n_kernel_rbf
  n_level_shape_ridge --> n_parrot
  n_lightgbm_tabular --> n_STOP
  n_lightgbm_tabular --> n_parrot
  n_lightgbm_tabular --> n_parrot_dataset
  n_parrot --> n_STOP
  n_parrot --> n_kernel_rbf
  n_parrot --> n_lightgbm_tabular
  n_parrot --> n_rf_tabular
  n_parrot --> n_versatile_gb
  n_parrot_dataset --> n_STOP
  n_parrot_dataset --> n_kernel_rbf
  n_parrot_dataset --> n_lightgbm_tabular
  n_parrot_dataset --> n_rf_tabular
  n_rf_tabular --> n_STOP
  n_rf_tabular --> n_parrot
  n_rf_tabular --> n_parrot_dataset
  n_step_regression --> n_STOP
  n_step_regression --> n_parrot
  n_versatile_gb --> n_STOP
  n_versatile_gb --> n_parrot
  n_versatile_gb --> n_versatile_gb
  classDef control fill:#ffe2e2,stroke:#b91c1c,stroke-width:1px,color:#111827
  classDef cleaning fill:#dff0ff,stroke:#2563eb,stroke-width:1px,color:#111827
  classDef transform fill:#fff1c2,stroke:#b45309,stroke-width:1px,color:#111827
  classDef feature fill:#d8f7ef,stroke:#047857,stroke-width:1px,color:#111827
  classDef binding fill:#efe3ff,stroke:#7c3aed,stroke-width:1px,color:#111827
  classDef model fill:#ffe6cf,stroke:#ea580c,stroke-width:1px,color:#111827
  classDef planned fill:#f4f4f5,stroke:#71717a,stroke-width:1px,color:#111827
  classDef terminal fill:#f0fdf4,stroke:#166534,stroke-width:1px,color:#111827
```

## Full Graph

Full view includes adapter/factory tokens and aliases.

```mermaid
flowchart LR
  n_START["START<br/>synthetic / control"]:::control
  n_FlairPreprocess["FlairPreprocess<br/>package / cleaning"]:::cleaning
  n_ForwardFill["ForwardFill<br/>package / cleaning"]:::cleaning
  n_LinearFill["LinearFill<br/>package / cleaning"]:::cleaning
  n_MeanAbsScaling["MeanAbsScaling<br/>package / transform"]:::transform
  n_ZNormalization["ZNormalization<br/>package / transform"]:::transform
  n_FlairSamplePaths["FlairSamplePaths<br/>package / feature"]:::feature
  n_FourierFeatures["FourierFeatures<br/>package / feature"]:::feature
  n_PeriodDetect["PeriodDetect<br/>package / feature"]:::feature
  n_PeriodDetectBIC["PeriodDetectBIC<br/>package / feature"]:::feature
  n_PeriodDetectSpectral["PeriodDetectSpectral<br/>package / feature"]:::feature
  n_PeriodPhaseOneHot["PeriodPhaseOneHot<br/>package / feature"]:::feature
  n_SeasonalFeatures["SeasonalFeatures<br/>package / feature"]:::feature
  n_SeasonalFold["SeasonalFold<br/>package / feature"]:::feature
  n_FlairRidgeLevel["FlairRidgeLevel<br/>package / model"]:::model
  n_gb_level_forecast["gb_level_forecast<br/>package / model"]:::model
  n_kernel_rbf["kernel_rbf<br/>package / model"]:::model
  n_level_shape_ridge["level_shape_ridge<br/>package / model"]:::model
  n_lightgbm_tabular["lightgbm_tabular<br/>package / model"]:::model
  n_parrot["parrot<br/>package / model"]:::model
  n_parrot_dataset["parrot_dataset<br/>package / model"]:::model
  n_rf_tabular["rf_tabular<br/>package / model"]:::model
  n_step_regression["step_regression<br/>package / model"]:::model
  n_versatile_gb["versatile_gb<br/>package / model"]:::model
  n_STOP["STOP<br/>synthetic / terminal"]:::terminal
  n_FlairPreprocess --> n_PeriodDetect
  n_FlairPreprocess --> n_PeriodDetectBIC
  n_FlairPreprocess --> n_PeriodDetectSpectral
  n_FlairRidgeLevel --> n_FlairSamplePaths
  n_FlairRidgeLevel --> n_STOP
  n_FlairRidgeLevel --> n_parrot
  n_FlairSamplePaths --> n_STOP
  n_ForwardFill --> n_FourierFeatures
  n_ForwardFill --> n_MeanAbsScaling
  n_ForwardFill --> n_PeriodDetect
  n_ForwardFill --> n_PeriodDetectBIC
  n_ForwardFill --> n_PeriodDetectSpectral
  n_ForwardFill --> n_ZNormalization
  n_ForwardFill --> n_kernel_rbf
  n_ForwardFill --> n_lightgbm_tabular
  n_ForwardFill --> n_parrot
  n_ForwardFill --> n_parrot_dataset
  n_ForwardFill --> n_rf_tabular
  n_ForwardFill --> n_versatile_gb
  n_FourierFeatures --> n_FourierFeatures
  n_FourierFeatures --> n_kernel_rbf
  n_FourierFeatures --> n_lightgbm_tabular
  n_FourierFeatures --> n_parrot
  n_FourierFeatures --> n_parrot_dataset
  n_FourierFeatures --> n_rf_tabular
  n_FourierFeatures --> n_step_regression
  n_FourierFeatures --> n_versatile_gb
  n_LinearFill --> n_FourierFeatures
  n_LinearFill --> n_MeanAbsScaling
  n_LinearFill --> n_PeriodDetect
  n_LinearFill --> n_PeriodDetectBIC
  n_LinearFill --> n_PeriodDetectSpectral
  n_LinearFill --> n_ZNormalization
  n_LinearFill --> n_kernel_rbf
  n_LinearFill --> n_lightgbm_tabular
  n_LinearFill --> n_parrot
  n_LinearFill --> n_parrot_dataset
  n_LinearFill --> n_rf_tabular
  n_LinearFill --> n_versatile_gb
  n_MeanAbsScaling --> n_FourierFeatures
  n_MeanAbsScaling --> n_PeriodDetect
  n_MeanAbsScaling --> n_PeriodDetectBIC
  n_MeanAbsScaling --> n_PeriodDetectSpectral
  n_MeanAbsScaling --> n_kernel_rbf
  n_MeanAbsScaling --> n_lightgbm_tabular
  n_MeanAbsScaling --> n_parrot
  n_MeanAbsScaling --> n_parrot_dataset
  n_MeanAbsScaling --> n_rf_tabular
  n_MeanAbsScaling --> n_step_regression
  n_MeanAbsScaling --> n_versatile_gb
  n_PeriodDetect --> n_PeriodPhaseOneHot
  n_PeriodDetect --> n_SeasonalFeatures
  n_PeriodDetect --> n_SeasonalFold
  n_PeriodDetect --> n_step_regression
  n_PeriodDetectBIC --> n_PeriodPhaseOneHot
  n_PeriodDetectBIC --> n_SeasonalFeatures
  n_PeriodDetectBIC --> n_SeasonalFold
  n_PeriodDetectBIC --> n_step_regression
  n_PeriodDetectSpectral --> n_PeriodPhaseOneHot
  n_PeriodDetectSpectral --> n_SeasonalFeatures
  n_PeriodDetectSpectral --> n_SeasonalFold
  n_PeriodDetectSpectral --> n_step_regression
  n_PeriodPhaseOneHot --> n_SeasonalFold
  n_START --> n_FlairPreprocess
  n_START --> n_ForwardFill
  n_START --> n_FourierFeatures
  n_START --> n_LinearFill
  n_START --> n_MeanAbsScaling
  n_START --> n_PeriodDetect
  n_START --> n_PeriodDetectBIC
  n_START --> n_PeriodDetectSpectral
  n_START --> n_ZNormalization
  n_START --> n_versatile_gb
  n_SeasonalFeatures --> n_kernel_rbf
  n_SeasonalFeatures --> n_lightgbm_tabular
  n_SeasonalFeatures --> n_parrot
  n_SeasonalFeatures --> n_rf_tabular
  n_SeasonalFeatures --> n_step_regression
  n_SeasonalFeatures --> n_versatile_gb
  n_SeasonalFold --> n_FlairRidgeLevel
  n_SeasonalFold --> n_SeasonalFold
  n_SeasonalFold --> n_gb_level_forecast
  n_SeasonalFold --> n_level_shape_ridge
  n_ZNormalization --> n_FourierFeatures
  n_ZNormalization --> n_PeriodDetect
  n_ZNormalization --> n_PeriodDetectBIC
  n_ZNormalization --> n_PeriodDetectSpectral
  n_ZNormalization --> n_kernel_rbf
  n_ZNormalization --> n_lightgbm_tabular
  n_ZNormalization --> n_parrot
  n_ZNormalization --> n_parrot_dataset
  n_ZNormalization --> n_rf_tabular
  n_ZNormalization --> n_step_regression
  n_ZNormalization --> n_versatile_gb
  n_gb_level_forecast --> n_STOP
  n_gb_level_forecast --> n_parrot
  n_kernel_rbf --> n_STOP
  n_kernel_rbf --> n_kernel_rbf
  n_kernel_rbf --> n_parrot
  n_kernel_rbf --> n_parrot_dataset
  n_level_shape_ridge --> n_STOP
  n_level_shape_ridge --> n_kernel_rbf
  n_level_shape_ridge --> n_parrot
  n_lightgbm_tabular --> n_STOP
  n_lightgbm_tabular --> n_parrot
  n_lightgbm_tabular --> n_parrot_dataset
  n_parrot --> n_STOP
  n_parrot --> n_kernel_rbf
  n_parrot --> n_lightgbm_tabular
  n_parrot --> n_rf_tabular
  n_parrot --> n_versatile_gb
  n_parrot_dataset --> n_STOP
  n_parrot_dataset --> n_kernel_rbf
  n_parrot_dataset --> n_lightgbm_tabular
  n_parrot_dataset --> n_rf_tabular
  n_rf_tabular --> n_STOP
  n_rf_tabular --> n_parrot
  n_rf_tabular --> n_parrot_dataset
  n_step_regression --> n_STOP
  n_step_regression --> n_parrot
  n_versatile_gb --> n_STOP
  n_versatile_gb --> n_parrot
  n_versatile_gb --> n_versatile_gb
  classDef control fill:#ffe2e2,stroke:#b91c1c,stroke-width:1px,color:#111827
  classDef cleaning fill:#dff0ff,stroke:#2563eb,stroke-width:1px,color:#111827
  classDef transform fill:#fff1c2,stroke:#b45309,stroke-width:1px,color:#111827
  classDef feature fill:#d8f7ef,stroke:#047857,stroke-width:1px,color:#111827
  classDef binding fill:#efe3ff,stroke:#7c3aed,stroke-width:1px,color:#111827
  classDef model fill:#ffe6cf,stroke:#ea580c,stroke-width:1px,color:#111827
  classDef planned fill:#f4f4f5,stroke:#71717a,stroke-width:1px,color:#111827
  classDef terminal fill:#f0fdf4,stroke:#166534,stroke-width:1px,color:#111827
```

## Token Matrix

| Token | Status | Class | Core | Parents | Next |
| --- | --- | --- | --- | --- | --- |
| `START` | synthetic | control | yes |  | `FlairPreprocess`, `ForwardFill`, `FourierFeatures`, `LinearFill`, `MeanAbsScaling`, `PeriodDetect`, `PeriodDetectBIC`, `PeriodDetectSpectral`, `ZNormalization`, `versatile_gb` |
| `FlairPreprocess` | package | cleaning | yes | `START` | `PeriodDetect`, `PeriodDetectBIC`, `PeriodDetectSpectral` |
| `ForwardFill` | package | cleaning | yes | `START` | `FourierFeatures`, `MeanAbsScaling`, `PeriodDetect`, `PeriodDetectBIC`, `PeriodDetectSpectral`, `ZNormalization`, `kernel_rbf`, `lightgbm_tabular`, `parrot`, `parrot_dataset`, `rf_tabular`, `versatile_gb` |
| `LinearFill` | package | cleaning | yes | `START` | `FourierFeatures`, `MeanAbsScaling`, `PeriodDetect`, `PeriodDetectBIC`, `PeriodDetectSpectral`, `ZNormalization`, `kernel_rbf`, `lightgbm_tabular`, `parrot`, `parrot_dataset`, `rf_tabular`, `versatile_gb` |
| `MeanAbsScaling` | package | transform | yes | `ForwardFill`, `LinearFill`, `START` | `FourierFeatures`, `PeriodDetect`, `PeriodDetectBIC`, `PeriodDetectSpectral`, `kernel_rbf`, `lightgbm_tabular`, `parrot`, `parrot_dataset`, `rf_tabular`, `step_regression`, `versatile_gb` |
| `ZNormalization` | package | transform | yes | `ForwardFill`, `LinearFill`, `START` | `FourierFeatures`, `PeriodDetect`, `PeriodDetectBIC`, `PeriodDetectSpectral`, `kernel_rbf`, `lightgbm_tabular`, `parrot`, `parrot_dataset`, `rf_tabular`, `step_regression`, `versatile_gb` |
| `FlairSamplePaths` | package | feature | yes | `FlairRidgeLevel` | `STOP` |
| `FourierFeatures` | package | feature | yes | `ForwardFill`, `FourierFeatures`, `LinearFill`, `MeanAbsScaling`, `START`, `ZNormalization` | `FourierFeatures`, `kernel_rbf`, `lightgbm_tabular`, `parrot`, `parrot_dataset`, `rf_tabular`, `step_regression`, `versatile_gb` |
| `PeriodDetect` | package | feature | yes | `FlairPreprocess`, `ForwardFill`, `LinearFill`, `MeanAbsScaling`, `START`, `ZNormalization` | `PeriodPhaseOneHot`, `SeasonalFeatures`, `SeasonalFold`, `step_regression` |
| `PeriodDetectBIC` | package | feature | yes | `FlairPreprocess`, `ForwardFill`, `LinearFill`, `MeanAbsScaling`, `START`, `ZNormalization` | `PeriodPhaseOneHot`, `SeasonalFeatures`, `SeasonalFold`, `step_regression` |
| `PeriodDetectSpectral` | package | feature | yes | `FlairPreprocess`, `ForwardFill`, `LinearFill`, `MeanAbsScaling`, `START`, `ZNormalization` | `PeriodPhaseOneHot`, `SeasonalFeatures`, `SeasonalFold`, `step_regression` |
| `PeriodPhaseOneHot` | package | feature | yes | `PeriodDetect`, `PeriodDetectBIC`, `PeriodDetectSpectral` | `SeasonalFold` |
| `SeasonalFeatures` | package | feature | yes | `PeriodDetect`, `PeriodDetectBIC`, `PeriodDetectSpectral` | `kernel_rbf`, `lightgbm_tabular`, `parrot`, `rf_tabular`, `step_regression`, `versatile_gb` |
| `SeasonalFold` | package | feature | yes | `PeriodDetect`, `PeriodDetectBIC`, `PeriodDetectSpectral`, `PeriodPhaseOneHot`, `SeasonalFold` | `FlairRidgeLevel`, `SeasonalFold`, `gb_level_forecast`, `level_shape_ridge` |
| `FlairRidgeLevel` | package | model | yes | `SeasonalFold` | `FlairSamplePaths`, `STOP`, `parrot` |
| `gb_level_forecast` | package | model | yes | `SeasonalFold` | `STOP`, `parrot` |
| `kernel_rbf` | package | model | yes | `ForwardFill`, `FourierFeatures`, `LinearFill`, `MeanAbsScaling`, `SeasonalFeatures`, `ZNormalization`, `kernel_rbf`, `level_shape_ridge`, `parrot`, `parrot_dataset` | `STOP`, `kernel_rbf`, `parrot`, `parrot_dataset` |
| `level_shape_ridge` | package | model | yes | `SeasonalFold` | `STOP`, `kernel_rbf`, `parrot` |
| `lightgbm_tabular` | package | model | yes | `ForwardFill`, `FourierFeatures`, `LinearFill`, `MeanAbsScaling`, `SeasonalFeatures`, `ZNormalization`, `parrot`, `parrot_dataset` | `STOP`, `parrot`, `parrot_dataset` |
| `parrot` | package | model | yes | `FlairRidgeLevel`, `ForwardFill`, `FourierFeatures`, `LinearFill`, `MeanAbsScaling`, `SeasonalFeatures`, `ZNormalization`, `gb_level_forecast`, `kernel_rbf`, `level_shape_ridge`, `lightgbm_tabular`, `rf_tabular`, `step_regression`, `versatile_gb` | `STOP`, `kernel_rbf`, `lightgbm_tabular`, `rf_tabular`, `versatile_gb` |
| `parrot_dataset` | package | model | yes | `ForwardFill`, `FourierFeatures`, `LinearFill`, `MeanAbsScaling`, `ZNormalization`, `kernel_rbf`, `lightgbm_tabular`, `rf_tabular` | `STOP`, `kernel_rbf`, `lightgbm_tabular`, `rf_tabular` |
| `rf_tabular` | package | model | yes | `ForwardFill`, `FourierFeatures`, `LinearFill`, `MeanAbsScaling`, `SeasonalFeatures`, `ZNormalization`, `parrot`, `parrot_dataset` | `STOP`, `parrot`, `parrot_dataset` |
| `step_regression` | package | model | yes | `FourierFeatures`, `MeanAbsScaling`, `PeriodDetect`, `PeriodDetectBIC`, `PeriodDetectSpectral`, `SeasonalFeatures`, `ZNormalization` | `STOP`, `parrot` |
| `versatile_gb` | package | model | yes | `ForwardFill`, `FourierFeatures`, `LinearFill`, `MeanAbsScaling`, `START`, `SeasonalFeatures`, `ZNormalization`, `parrot`, `versatile_gb` | `STOP`, `parrot`, `versatile_gb` |
| `STOP` | synthetic | terminal | yes | `FlairRidgeLevel`, `FlairSamplePaths`, `gb_level_forecast`, `kernel_rbf`, `level_shape_ridge`, `lightgbm_tabular`, `parrot`, `parrot_dataset`, `rf_tabular`, `step_regression`, `versatile_gb` |  |
