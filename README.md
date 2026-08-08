# China Provincial Energy Consumption and Sustainable Development Indicators Panel Dataset (1990–2025)

## Overview

This repository provides the data, documentation and analytical code supporting the manuscript:

**“Data-driven forecasting of energy consumption in China and implications for sustainable development policy.”**

The dataset integrates provincial-level economic, industrial, energy-supply, environmental and carbon-emission indicators for 31 provincial-level regions in mainland China from 1990 to 2025. It was compiled to support machine-learning analysis of provincial total energy consumption and its associated socioeconomic, energy and emissions characteristics.

## Repository structure

```text
02_Data/
├── Supplementary_Data.xlsx
├── DATA_AVAILABILITY.md
├── README.md
└── csv/
    ├── raw_data.csv
    ├── imputed_all_rows.csv
    ├── analysis_data.csv
    ├── imputation_flags.csv
    ├── missingness_report.csv
    ├── variable_dictionary.csv
    ├── sources_and_methods.csv
    ├── data_qa.csv
    └── province_region_mapping.csv

03_Code/
├── model_analysis.py
├── prepare_data.py
├── verify_reproducibility.py
├── export_data_sheets.py
├── requirements.txt
├── run_analysis.bat
├── METHODS_AND_FORMULAE.md
└── README.md
```

## Dataset summary

- Unit of observation: province-year
- Spatial coverage: 31 provincial-level regions in mainland China
- Temporal coverage of the compiled panel: 1990–2025
- Raw panel: 1,116 province-year observations
- Model-ready sample: 1,003 observations
- Missing total energy consumption records: 113
- Prediction target: total energy consumption
- Final model inputs: 14
- Province identifiers are retained for traceability but are not used as model inputs
- Region is included as a categorical model input

The availability of observed total energy consumption varies across years. In particular, the inclusion of 2025 in the raw panel does not imply that complete outcome observations are available for every province in that year.

## Data files

### Supplementary_Data.xlsx

The Excel workbook is the authoritative formatted data package and contains the following sheets:

- `Raw_Data`: the complete 1,116-row panel before predictor imputation
- `Imputed_All_Rows`: all panel rows after predictor imputation, with the outcome left unimputed
- `Analysis_Data`: the 1,003-row dataset used for supervised learning
- `Imputation_Flags`: row-level indicators identifying imputed predictor values
- `Variable_Dictionary`: variable names, abbreviations, roles, units, definitions and sources
- `Missingness_Report`: missingness and imputation counts
- `Sources_and_Methods`: data provenance and processing rules
- `Data_QA`: data-quality and reproducibility checks

### CSV files

Open UTF-8 CSV versions of the principal workbook sheets are provided in `02_Data/csv/`.

Province and region values are retained in their original Chinese form to preserve exact consistency with the analysis data. English translations are provided in `province_region_mapping.csv`.

## Variables

The prediction target is provincial total energy consumption. The 14 final inputs describe:

- observation year and macro-region
- regional gross domestic product
- secondary- and tertiary-industry value added
- industrial value added
- electricity generation
- population
- hydropower generation
- raw coal production
- non-fossil electricity generation
- sulphur dioxide emissions
- natural gas production
- apparent carbon dioxide emissions

Complete definitions, units and source mappings are provided in the `Variable_Dictionary` sheet and `variable_dictionary.csv`.

## Data sources

The compiled panel draws primarily on:

- China Statistical Yearbook
- China Energy Statistical Yearbook
- China Statistical Yearbook on Environment
- relevant provincial statistical yearbooks
- China Emission Accounts and Datasets provincial apparent CO2 emission inventory

The original source publications and datasets remain subject to the terms and citation requirements of their respective providers.

## Data processing

The total energy consumption outcome was never imputed. Rows with missing outcome values were excluded from supervised learning.

Missing predictor values were processed in the following order:

1. linear interpolation over time within each province;
2. forward and backward filling within each province;
3. region-by-year median filling;
4. overall median filling when earlier steps were insufficient.

The `Imputation_Flags` and `Missingness_Report` files provide a complete audit trail of this process.

## Analytical workflow

The supplied code compares six tree-based ensemble models:

- AdaBoost
- XGBoost
- random forest
- gradient boosting decision tree
- CatBoost
- LightGBM

Hyperparameters are optimised using Optuna and five-fold cross-validation. The reported workflow uses an 80%/20% randomly shuffled train/test split with random seed 42. The outcome is transformed using `log1p` during model fitting and returned to its original scale using `expm1` for evaluation.

SHAP and partial-dependence analyses are implemented to examine predictive feature contributions and nonlinear response patterns. These analyses describe model-based statistical associations and should not be interpreted as evidence of causal effects.

The repository reproduces the random-split workflow reported in the manuscript. Temporal hold-out, leave-province-out and causal analyses are outside the scope of the supplied code.

## Reproducibility

Install the required Python packages:

```text
python -m pip install -r 03_Code/requirements.txt
```

Verify that the model-ready dataset can be reconstructed from the raw data:

```text
python 03_Code/verify_reproducibility.py
```

Expected output:

```text
PASS: 1,003 analysis rows, 14 inputs and all submitted values reproduce exactly.
```

Run the complete modelling workflow:

```text
python 03_Code/model_analysis.py
```

The complete analysis performs 600 Optuna trials and may require substantial computation time.

## Data Availability

The raw provincial panel data, processed model-ready data, imputation records, missingness report, variable definitions, source documentation, data-processing scripts, statistical analysis code, formulae, software requirements and reproducibility instructions supporting this study are publicly available in this repository.

The underlying statistical yearbooks and CEADs source data remain subject to the terms of their respective providers.

## Citation

When using these data or code, please cite the associated manuscript and the final repository release:

Li, H., Li, W., Zhao, G., Wang, N., Wu, J. and Wang, M. *Data-driven forecasting of energy consumption in China and implications for sustainable development policy*. Manuscript under review.

A permanent repository DOI should be added here after the GitHub release is archived.
