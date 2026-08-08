"""Validate the submitted analysis table against the documented preparation steps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from prepare_data import build_analysis_data


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR.parent / "02_Data" / "Supplementary_Data.xlsx"


def main() -> None:
    raw = pd.read_excel(DATA_FILE, sheet_name="Raw_Data")
    supplied = pd.read_excel(DATA_FILE, sheet_name="Analysis_Data")
    recreated = build_analysis_data(raw)

    if list(supplied.columns) != list(recreated.columns):
        raise AssertionError("Column order differs between supplied and recreated analysis data.")
    if supplied.shape != (1003, 16):
        raise AssertionError(f"Unexpected supplied shape: {supplied.shape}; expected (1003, 16).")

    categorical = [column for column in supplied.columns if not is_numeric_dtype(supplied[column])]
    numerical = [column for column in supplied.columns if column not in categorical]
    for column in categorical:
        if not supplied[column].fillna("<NA>").equals(recreated[column].fillna("<NA>")):
            raise AssertionError(f"Categorical mismatch in {column}.")
    if not np.allclose(
        supplied[numerical].to_numpy(float),
        recreated[numerical].to_numpy(float),
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    ):
        raise AssertionError("Numerical values differ between supplied and recreated analysis data.")

    observed_target = raw["Total energy consumption"].notna().sum()
    if observed_target != 1003:
        raise AssertionError(f"Observed target count is {observed_target}; expected 1003.")

    print("PASS: 1,003 analysis rows, 14 inputs and all submitted values reproduce exactly.")


if __name__ == "__main__":
    main()
