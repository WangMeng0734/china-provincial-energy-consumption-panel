"""Export the principal workbook sheets to UTF-8 CSV files for open inspection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORKBOOK = SCRIPT_DIR.parent / "02_Data" / "Supplementary_Data.xlsx"
OUTPUT_DIR = SCRIPT_DIR.parent / "02_Data" / "csv"

SHEETS = {
    "Raw_Data": "raw_data.csv",
    "Imputed_All_Rows": "imputed_all_rows.csv",
    "Analysis_Data": "analysis_data.csv",
    "Imputation_Flags": "imputation_flags.csv",
    "Variable_Dictionary": "variable_dictionary.csv",
    "Missingness_Report": "missingness_report.csv",
    "Sources_and_Methods": "sources_and_methods.csv",
    "Data_QA": "data_qa.csv",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for sheet_name, filename in SHEETS.items():
        frame = pd.read_excel(WORKBOOK, sheet_name=sheet_name)
        output_path = OUTPUT_DIR / filename
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Exported {sheet_name}: {frame.shape[0]} rows x {frame.shape[1]} columns -> {output_path}")


if __name__ == "__main__":
    main()
