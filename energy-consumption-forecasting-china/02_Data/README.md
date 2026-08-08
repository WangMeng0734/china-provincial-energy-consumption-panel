# Data files

`Supplementary_Data.xlsx` is the authoritative formatted workbook. The `csv/` directory contains UTF-8 exports for software-independent inspection.

## Workbook sheets and CSV equivalents

| Workbook sheet | CSV file | Purpose |
|---|---|---|
| `Raw_Data` | `csv/raw_data.csv` | Full 1,116-row panel before predictor imputation; TEC missingness is retained. |
| `Imputed_All_Rows` | `csv/imputed_all_rows.csv` | All 1,116 rows after predictor imputation; TEC remains unimputed. |
| `Analysis_Data` | `csv/analysis_data.csv` | The 1,003-row model-ready dataset used in supervised learning. |
| `Imputation_Flags` | `csv/imputation_flags.csv` | Boolean audit trail for every imputed predictor value. |
| `Variable_Dictionary` | `csv/variable_dictionary.csv` | Column names, abbreviations, roles, units, definitions, and sources. |
| `Missingness_Report` | `csv/missingness_report.csv` | Missingness before and after processing and imputation counts. |
| `Sources_and_Methods` | `csv/sources_and_methods.csv` | Source families, coverage, processing rules, and outcome handling. |
| `Data_QA` | `csv/data_qa.csv` | Dataset-level quality and reproducibility checks. |

The workbook also includes a formatted `README` sheet.

Province and region labels are retained in their original Chinese form so that the shared files remain identical to the data used in the analysis. `csv/province_region_mapping.csv` provides the corresponding English names for international reviewers.

Run `python 03_Code/export_data_sheets.py` from the repository root to recreate the CSV exports.
