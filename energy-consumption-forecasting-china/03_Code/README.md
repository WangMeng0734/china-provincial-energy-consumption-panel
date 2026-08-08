# Supplementary code

This package supports the manuscript **“Data-driven forecasting of energy consumption in China and implications for sustainable development policy”**.

## Contents

- `model_analysis.py`: six-model Optuna comparison, evaluation, SHAP analysis and partial-dependence analysis.
- `prepare_data.py`: recreates the model-ready table from the `Raw_Data` sheet.
- `verify_reproducibility.py`: checks that the supplied `Analysis_Data` sheet is reproduced exactly.
- `export_data_sheets.py`: recreates the open CSV versions of the workbook sheets.
- `METHODS_AND_FORMULAE.md`: documents the transformations, validation settings and statistical formulae implemented in the analysis.
- `requirements.txt`: Python dependencies.

The code expects `../02_Data/Supplementary_Data.xlsx`. It does not overwrite that workbook. Generated files are written to `outputs/` or, for the preparation script, to `recreated_analysis_data.xlsx` in this folder.

## Reproducibility settings

- Random seed: 42.
- Train/test split: 80%/20%, randomly shuffled.
- Five-fold cross-validation on the training set.
- Target transformation: `log1p`; predictions returned using `expm1`.
- Optuna sampler: tree-structured Parzen estimator with 10 start-up trials and 100 total trials for each model.
- Province is retained for tracing but excluded from model inputs.
- Region is one-hot encoded.
- Per-capita GDP is excluded before modelling.

## Run

Create a clean Python environment, install the dependencies, then run:

```text
python verify_reproducibility.py
python model_analysis.py
```

The full optimisation performs 600 Optuna trials in total and may require substantial computation time. Optional model libraries must be installed for all six candidate models to run.

## Data-processing sequence

For predictor variables only, missing values are filled in this order: province-level linear interpolation over time; province-level forward and backward fill; region-by-year median; overall median. The TEC outcome is never imputed, and rows with missing TEC are excluded from supervised learning.

## Reporting boundary

The submitted code reproduces the reported random train/test modelling workflow. Temporal hold-out, leave-province-out and feature-ablation analyses are not represented as completed experiments in the manuscript and are therefore not included here.
