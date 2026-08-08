"""Recreate the model-ready analysis table from Supplementary_Data.xlsx."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = SCRIPT_DIR.parent / "02_Data" / "Supplementary_Data.xlsx"
TARGET = "Total energy consumption"
EXCLUDED_INPUTS = {"GDP per capita from Long"}


def build_analysis_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the imputation sequence described in the manuscript."""
    data = raw.copy()
    data["__source_order"] = np.arange(len(data))
    data = data.sort_values(["Province", "Year"], kind="stable").reset_index(drop=True)
    predictors = [
        column
        for column in data.columns
        if column not in {"Province", TARGET, *EXCLUDED_INPUTS}
    ]
    numerical_predictors = [column for column in predictors if column != "Region"]

    for column in numerical_predictors:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data.groupby("Province", sort=False)[column].transform(
            lambda series: series.interpolate(method="linear").ffill().bfill()
        )
        regional_year_median = data.groupby(["Region", "Year"], dropna=False)[column].transform("median")
        data[column] = data[column].fillna(regional_year_median)
        data[column] = data[column].fillna(data[column].median())

    data = data.sort_values("__source_order", kind="stable")
    final_columns = [column for column in raw.columns if column not in EXCLUDED_INPUTS]
    data = data[final_columns]
    return data.dropna(subset=[TARGET]).reset_index(drop=True)


def main() -> None:
    raw = pd.read_excel(DEFAULT_DATA_FILE, sheet_name="Raw_Data")
    analysis = build_analysis_data(raw)
    output = SCRIPT_DIR / "recreated_analysis_data.xlsx"
    analysis.to_excel(output, sheet_name="Analysis_Data", index=False)
    print(f"Wrote {len(analysis):,} model-ready rows with {len(analysis.columns) - 2:,} predictors to {output}")


if __name__ == "__main__":
    main()
