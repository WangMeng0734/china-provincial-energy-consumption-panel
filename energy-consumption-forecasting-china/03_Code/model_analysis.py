# -*- coding: utf-8 -*-
"""Reproduce the six-model Optuna comparison and the SHAP/PDP analyses.

The workflow reads the submitted model-ready panel, compares AdaBoost, XGBoost,
random forest, GBDT, CatBoost and LightGBM, and writes all generated outputs to
the local ``outputs`` directory. See README.md and METHODS_AND_FORMULAE.md for
the data-processing sequence, statistical definitions and reporting boundary.
"""

import os
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from packaging import version
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    explained_variance_score,
)
from sklearn.inspection import partial_dependence
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.base import clone

try:
    import optuna
    from optuna.samplers import TPESampler
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing required package: optuna. Please install it with: pip install optuna"
    ) from e

# Optional models
try:
    from xgboost import XGBRegressor
except Exception as e:
    XGBRegressor = None
    print("Warning: xgboost is not available:", e)

try:
    from lightgbm import LGBMRegressor
except Exception as e:
    LGBMRegressor = None
    print("Warning: lightgbm is not available:", e)

try:
    from catboost import CatBoostRegressor
except Exception as e:
    CatBoostRegressor = None
    print("Warning: catboost is not available:", e)

try:
    import shap
except Exception as e:
    shap = None
    print("Warning: shap is not available:", e)


# =========================

# =========================

RANDOM_STATE = 42


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent / "02_Data" / "Supplementary_Data.xlsx"
SHEET_NAME = "Analysis_Data"

TARGET_COL = "Total energy consumption"
PROVINCE_COL = "Province"
REGION_COL = "Region"


DROP_FEATURE_COLS = [
    "GDP per capita from Long",
    "GDP per capita",
    "Per capita GDP",
    "GDPpc",
]


TEST_SIZE = 0.20


N_STARTUP_TRIALS = 10   
N_TRIALS = 100         


N_SPLITS = 5


OUT_DIR = SCRIPT_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
SHAP_DIR = OUT_DIR / "shap_results"
PRED_SCATTER_DIR = OUT_DIR / "prediction_scatter"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR.mkdir(parents=True, exist_ok=True)
PRED_SCATTER_DIR.mkdir(parents=True, exist_ok=True)


CM_TO_INCH = 1 / 2.54
FIG_DPI = 800  



PARALLEL_FIG_WIDTH_CM = 8.0
PARALLEL_FIG_HEIGHT_CM = 6.2


MIN_FIG_WIDTH_CM = 10.0
MAX_FIG_WIDTH_CM = 18.0
BASE_HEIGHT_CM = 7.0
MAX_FIG_HEIGHT_CM = 16.0


COMMON_CMAP_NAME = "rainbow"
COMMON_CMAP = plt.cm.get_cmap(COMMON_CMAP_NAME)

# =========================

# =========================


SHAP_COMBINED_BAR_COLOR = "#F4B183"
SHAP_COMBINED_BAR_ALPHA = 0.42


PRED_SCATTER_FACE_COLOR = "white"


SHAP_DEPENDENCE_WIDTH_CM = 8.0
SHAP_DEPENDENCE_HEIGHT_CM = 6.2

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["mathtext.fontset"] = "stix"


def apply_times_new_roman(fig=None, font_size=11):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    if fig is None:
        fig = plt.gcf()

    for ax in fig.axes:
        ax.title.set_fontname("Times New Roman")
        ax.title.set_fontsize(font_size)
        ax.xaxis.label.set_fontname("Times New Roman")
        ax.xaxis.label.set_fontsize(font_size)
        ax.yaxis.label.set_fontname("Times New Roman")
        ax.yaxis.label.set_fontsize(font_size)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(font_size)

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontname("Times New Roman")
                text.set_fontsize(font_size)

        for text in ax.texts:
            text.set_fontname("Times New Roman")
            text.set_fontsize(font_size)

    
    for text in fig.findobj(match=matplotlib.text.Text):
        text.set_fontname("Times New Roman")
        text.set_fontsize(font_size)




def save_figure_dual(fig, save_path, dpi=FIG_DPI, bbox_inches="tight"):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    svg_path = save_path.with_suffix(".svg")
    png_path = save_path.with_suffix(".png")

    
    apply_times_new_roman(fig, font_size=11)

    
    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches=bbox_inches)

    
    old_svg_fonttype = plt.rcParams.get("svg.fonttype", "path")
    try:
        plt.rcParams["svg.fonttype"] = "path"
        fig.savefig(svg_path, format="svg", bbox_inches=bbox_inches)
    finally:
        plt.rcParams["svg.fonttype"] = old_svg_fonttype

    print("Saved PNG to:", png_path)
    print("Saved SVG to:", svg_path)
    return svg_path, png_path



def format_plot_label(label):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    if label is None:
        return label
    text = str(label)
    text = text.replace("CO2", "CO₂")
    text = text.replace("SO2_alt", "SO₂-alt")
    text = text.replace("SO2", "SO₂")
    return text

def smooth_parallel_curve(x, y, points_per_interval=40):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        x_new = np.linspace(x.min(), x.max(), max(50, len(x) * points_per_interval))
        y_new = np.interp(x_new, x, y)
        return x_new, y_new

    x_new = np.linspace(x.min(), x.max(), max(120, (len(x) - 1) * points_per_interval))
    try:
        from scipy.interpolate import PchipInterpolator
        y_new = PchipInterpolator(x, y)(x_new)
    except Exception:
        y_new = np.interp(x_new, x, y)

    
    y_new = np.clip(y_new, 0.0, 1.0)
    return x_new, y_new


# =========================

# =========================


FEATURE_META = {
    "Year": {
        "abbr": "Year",
        "unit": "year",
        "meaning": "Observation year."
    },
    "Region": {
        "abbr": "Region",
        "unit": "category",
        "meaning": "Regional category of each province; one-hot encoded for modelling and merged for SHAP visualization."
    },
    "Regional GDP": {
        "abbr": "GDP",
        "unit": "10^8 CNY",
        "meaning": "Regional gross domestic product, reflecting the overall economic scale."
    },
    "Secondary industry value added": {
        "abbr": "SIVA",
        "unit": "10^8 CNY",
        "meaning": "Value added of the secondary industry, representing industrial and construction activity."
    },
    "Tertiary industry value added from Long": {
        "abbr": "TIVA",
        "unit": "10^8 CNY",
        "meaning": "Value added of the tertiary industry, representing service-sector development."
    },
    "GDP per capita from Long": {
        "abbr": "GDPpc",
        "unit": "CNY person^-1",
        "meaning": "Per-capita GDP, indicating the level of economic development."
    },
    "Industrial value added from Long": {
        "abbr": "IVA",
        "unit": "10^8 CNY",
        "meaning": "Industrial value added, reflecting the scale of industrial production."
    },
    "Total electricity generation": {
        "abbr": "TEG",
        "unit": "10^8 kWh",
        "meaning": "Total electricity generation, representing power supply capacity."
    },
    "Population": {
        "abbr": "POP",
        "unit": "10^4 persons",
        "meaning": "Population size, reflecting demographic demand for energy consumption."
    },
    "Hydropower generation from Long": {
        "abbr": "HPG",
        "unit": "10^8 kWh",
        "meaning": "Hydropower generation, representing renewable electricity output from hydropower."
    },
    "Raw coal production from Long": {
        "abbr": "RCP",
        "unit": "10^4 t",
        "meaning": "Raw coal production, indicating coal resource supply and fossil-energy dependence."
    },
    "Non fossil electricity generation": {
        "abbr": "NFEG",
        "unit": "10^8 kWh",
        "meaning": "Non-fossil electricity generation, reflecting low-carbon power supply."
    },
    "SO2 emissions": {
        "abbr": "SO₂",
        "unit": "10^4 t",
        "meaning": "Sulfur dioxide emissions, representing air-pollution intensity related to fossil-fuel use."
    },
    "Natural gas production from Long": {
        "abbr": "NGP",
        "unit": "10^8 m^3",
        "meaning": "Natural gas production, indicating natural gas supply capacity."
    },
    "Total apparent CO2 emissions (mt)": {
        "abbr": "CO₂",
        "unit": "Mt",
        "meaning": "Total apparent carbon dioxide emissions."
    },
    "Raw coal total": {
        "abbr": "RCT",
        "unit": "Mtce or source unit",
        "meaning": "Total raw coal consumption or apparent coal-related energy input, depending on the source table."
    },
    "Crude oil total": {
        "abbr": "COT",
        "unit": "Mtce or source unit",
        "meaning": "Total crude oil consumption or apparent oil-related energy input, depending on the source table."
    },
    "Natural gas total": {
        "abbr": "NGT",
        "unit": "Mtce or source unit",
        "meaning": "Total natural gas consumption or apparent gas-related energy input, depending on the source table."
    },
}


def simplify_feature_name(feature_name):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    feature_name = str(feature_name)

    # One-hot encoded region variables, e.g., Region_Eastern, Region_Western
    if feature_name == REGION_COL or feature_name.startswith(REGION_COL + "_"):
        return FEATURE_META.get(REGION_COL, {}).get("abbr", REGION_COL)

    if feature_name in FEATURE_META:
        return FEATURE_META[feature_name]["abbr"]

    
    cleaned = feature_name.replace("num__", "").replace("cat__", "")
    if cleaned == REGION_COL or cleaned.startswith(REGION_COL + "_"):
        return FEATURE_META.get(REGION_COL, {}).get("abbr", REGION_COL)
    if cleaned in FEATURE_META:
        return FEATURE_META[cleaned]["abbr"]

    
    cleaned = cleaned.replace(" from Long", "")
    return cleaned


def make_unique_feature_names(names):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    seen = {}
    out = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
    return out


def save_feature_metadata_table(input_columns, output_dir):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    rows = []
    for col in input_columns:
        if col == PROVINCE_COL or col == TARGET_COL:
            continue
        meta = FEATURE_META.get(col, None)
        if meta is None:
            rows.append({
                "Full name": col,
                "Unit": "source unit",
                "Abbreviation": simplify_feature_name(col),
                "Meaning": "Predictor variable included in the modelling dataset. Please update the meaning according to the source definition."
            })
        else:
            rows.append({
                "Full name": col,
                "Unit": meta["unit"],
                "Abbreviation": meta["abbr"],
                "Meaning": meta["meaning"]
            })

    meta_df = pd.DataFrame(rows)
    meta_path = Path(output_dir) / "Feature_abbreviation_unit_meaning_table.xlsx"
    meta_df.to_excel(meta_path, index=False)

    print("\nFeature abbreviation, unit, and meaning table:")
    print(meta_df.to_string(index=False))
    print("Saved feature metadata table to:", meta_path)
    return meta_df


def merge_region_shap_values(shap_values, X_exp_processed, feature_names):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    shap_array = np.asarray(shap_values)
    feature_names = list(feature_names)

    region_cols = [
        f for f in feature_names
        if f == REGION_COL or f.startswith(REGION_COL + "_") or f.startswith("cat__" + REGION_COL + "_")
    ]
    region_idx = [feature_names.index(f) for f in region_cols]
    other_cols = [f for f in feature_names if f not in region_cols]
    other_idx = [feature_names.index(f) for f in other_cols]

    if len(region_idx) == 0:
        merged_values = shap_array
        merged_X = X_exp_processed.copy()
        merged_names_raw = feature_names
    else:
        other_values = shap_array[:, other_idx]
        region_values = shap_array[:, region_idx].sum(axis=1, keepdims=True)
        merged_values = np.hstack([other_values, region_values])

        merged_X = X_exp_processed[other_cols].copy()
        
        region_matrix = X_exp_processed[region_cols].values
        if region_matrix.ndim == 2 and region_matrix.shape[1] > 0:
            merged_X[REGION_COL] = np.argmax(region_matrix, axis=1).astype(float)
        else:
            merged_X[REGION_COL] = 0.0
        merged_names_raw = other_cols + [REGION_COL]

    short_names = make_unique_feature_names([simplify_feature_name(f) for f in merged_names_raw])
    merged_X = merged_X.copy()
    merged_X.columns = short_names

    return merged_values, merged_X, short_names, merged_names_raw



# =========================

# =========================

FEATURE_META = {
    "Year": {
        "abbr": "Year",
        "unit": "year",
        "meaning": "Observation year"
    },
    "Region": {
        "abbr": "Region",
        "unit": "category",
        "meaning": "Regional category of each province"
    },
    "Regional GDP": {
        "abbr": "GDP",
        "unit": "10^8 CNY",
        "meaning": "Regional gross domestic product"
    },
    "Secondary industry value added": {
        "abbr": "SIVA",
        "unit": "10^8 CNY",
        "meaning": "Value added of the secondary industry"
    },
    "Tertiary industry value added from Long": {
        "abbr": "TIVA",
        "unit": "10^8 CNY",
        "meaning": "Value added of the tertiary industry"
    },
    "GDP per capita from Long": {
        "abbr": "GDPpc",
        "unit": "CNY person^-1",
        "meaning": "Per-capita GDP"
    },
    "Industrial value added from Long": {
        "abbr": "IVA",
        "unit": "10^8 CNY",
        "meaning": "Industrial value added"
    },
    "Total electricity generation": {
        "abbr": "TEG",
        "unit": "10^8 kWh",
        "meaning": "Total electricity generation"
    },
    "Population": {
        "abbr": "POP",
        "unit": "10^4 persons",
        "meaning": "Population size"
    },
    "Hydropower generation from Long": {
        "abbr": "HPG",
        "unit": "10^8 kWh",
        "meaning": "Hydropower generation"
    },
    "Raw coal production from Long": {
        "abbr": "RCP",
        "unit": "10^4 t",
        "meaning": "Raw coal production"
    },
    "Non fossil electricity generation": {
        "abbr": "NFEG",
        "unit": "10^8 kWh",
        "meaning": "Non-fossil electricity generation"
    },
    "SO2 emissions": {
        "abbr": "SO₂",
        "unit": "10^4 t",
        "meaning": "Sulfur dioxide emissions"
    },
    "SO2 emissions alternative from Long": {
        "abbr": "SO₂-alt",
        "unit": "10^4 t",
        "meaning": "Alternative sulfur dioxide emissions indicator"
    },
    "Natural gas production from Long": {
        "abbr": "NGP",
        "unit": "10^8 m^3",
        "meaning": "Natural gas production"
    },
    "Total apparent CO2 emissions (mt)": {
        "abbr": "CO₂",
        "unit": "Mt",
        "meaning": "Total apparent carbon dioxide emissions"
    },
    "Raw coal total": {
        "abbr": "RCT",
        "unit": "source unit",
        "meaning": "Total raw coal-related indicator from carbon emission inventory"
    },
    "Crude oil total": {
        "abbr": "COT",
        "unit": "source unit",
        "meaning": "Total crude-oil-related indicator from carbon emission inventory"
    },
    "Natural gas total": {
        "abbr": "NGT",
        "unit": "source unit",
        "meaning": "Total natural-gas-related indicator from carbon emission inventory"
    },
    "Total energy consumption": {
        "abbr": "TEC",
        "unit": "10^4 tce",
        "meaning": "Total energy consumption"
    },
}


def abbr_feature_name(feature_name):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    feature_name = str(feature_name)

    # Region one-hot features, e.g., Region_Eastern / Region_Western
    if feature_name == "Region":
        return "Region"
    if feature_name.startswith("Region_"):
        region_value = feature_name.replace("Region_", "")
        region_map = {
            "Eastern": "Reg_E",
            "Central": "Reg_C",
            "Western": "Reg_W",
            "Northeastern": "Reg_NE",
            "Northeast": "Reg_NE",
            "East": "Reg_E",
            "Middle": "Reg_C",
            "West": "Reg_W",
        }
        return region_map.get(region_value, "Reg_" + region_value[:6])

    if feature_name in FEATURE_META:
        return FEATURE_META[feature_name]["abbr"]

    # Fallback: create a compact abbreviation from words
    cleaned = (
        feature_name.replace("(", " ")
        .replace(")", " ")
        .replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
    )
    words = [w for w in cleaned.split() if len(w) > 0]
    if len(words) == 0:
        return feature_name
    if len(words) == 1:
        return words[0][:10]
    abbr = "".join([w[0].upper() for w in words if w.lower() not in {"from", "the", "and", "of"}])
    return abbr[:12] if abbr else feature_name[:12]


def save_feature_abbreviation_table(columns, out_dir):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seen = set()

    for col in columns:
        col = str(col)
        if col in seen:
            continue
        seen.add(col)

        if col.startswith("Region_"):
            full_name = col
            unit = "dummy variable"
            meaning = f"One-hot encoded regional category: {col.replace('Region_', '')}"
            abbr = abbr_feature_name(col)
        elif col in FEATURE_META:
            full_name = col
            unit = FEATURE_META[col]["unit"]
            meaning = FEATURE_META[col]["meaning"]
            abbr = FEATURE_META[col]["abbr"]
        else:
            full_name = col
            unit = "unknown"
            meaning = "User-defined or derived model input feature"
            abbr = abbr_feature_name(col)

        rows.append({
            "Full name": full_name,
            "Unit": unit,
            "Abbreviation": abbr,
            "Meaning": meaning
        })

    table = pd.DataFrame(rows)
    path = out_dir / "Feature_abbreviation_unit_meaning_table.xlsx"
    table.to_excel(path, index=False)
    print("Saved feature abbreviation table to:", path)
    print("\nFeature abbreviation table:")
    print(table)
    return table


def get_abbr_columns(columns):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    abbrs = []
    counts = {}
    for c in columns:
        a = abbr_feature_name(c)
        counts[a] = counts.get(a, 0) + 1
        if counts[a] > 1:
            a = f"{a}_{counts[a]}"
        abbrs.append(format_plot_label(a))
    return abbrs


def merge_region_shap_values(shap_values, X_processed, feature_names):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    shap_arr = np.asarray(shap_values)
    X_proc = X_processed.copy()
    names = list(feature_names)

    region_cols = [c for c in names if str(c).startswith("Region_")]

    if len(region_cols) == 0:
        
        return shap_arr, X_proc, names

    region_idx = [names.index(c) for c in region_cols]
    non_region_cols = [c for c in names if c not in region_cols]
    non_region_idx = [names.index(c) for c in non_region_cols]

    region_shap = shap_arr[:, region_idx].sum(axis=1)
    region_value = X_proc[region_cols].sum(axis=1).values

    merged_values = np.column_stack([shap_arr[:, non_region_idx], region_shap])
    merged_X = X_proc[non_region_cols].copy()
    merged_X["Region"] = region_value

    merged_names = non_region_cols + ["Region"]

    return merged_values, merged_X, merged_names


# =========================

# =========================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_regression(y_true, y_pred):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    abs_residual = np.abs(residual)

    mse_value = mean_squared_error(y_true, y_pred)
    rmse_value = np.sqrt(mse_value)
    mae_value = mean_absolute_error(y_true, y_pred)
    medae_value = median_absolute_error(y_true, y_pred)
    r2_value = r2_score(y_true, y_pred)
    evs_value = explained_variance_score(y_true, y_pred)

    
    eps = 1e-12
    mape_value = np.mean(abs_residual / np.maximum(np.abs(y_true), eps)) * 100.0
    smape_value = np.mean(2.0 * abs_residual / np.maximum(np.abs(y_true) + np.abs(y_pred), eps)) * 100.0

    
    sd_y = np.std(y_true, ddof=1) if len(y_true) > 1 else np.nan
    rpd_value = sd_y / rmse_value if rmse_value > 0 else np.nan
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    nse_value = 1.0 - np.sum(residual ** 2) / denom if denom > 0 else np.nan

    return {
        "R2": r2_value,
        "RMSE": rmse_value,
        "MSE": mse_value,
        "MAE": mae_value,
        "MedAE": medae_value,
        "MAPE_percent": mape_value,
        "sMAPE_percent": smape_value,
        "Explained_variance": evs_value,
        "RPD": rpd_value,
        "NSE": nse_value,
    }


def safe_expm1(x):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    x = np.asarray(x)
    x = np.clip(x, -20, 25)
    return np.expm1(x)


def make_onehot_encoder():
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(X, scale_numeric=False):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    if scale_numeric:
        numeric_transformer = StandardScaler()
    else:
        numeric_transformer = "passthrough"

    transformers = []
    if len(numeric_cols) > 0:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if len(categorical_cols) > 0:
        transformers.append(("cat", make_onehot_encoder(), categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_cols, categorical_cols


def get_feature_names(preprocessor):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return [f"feature_{i}" for i in range(preprocessor.transformers_[0][1].shape[1])]


def objective_cv_score(model, X_train, y_train_log, scale_numeric=False):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    preprocessor, _, _ = make_preprocessor(X_train, scale_numeric=scale_numeric)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipe,
        X_train,
        y_train_log,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        error_score="raise"
    )

    mean_cv_rmse = -scores.mean()
    return mean_cv_rmse


def run_optuna(model_name, objective_func):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    sampler = TPESampler(
        n_startup_trials=N_STARTUP_TRIALS,
        seed=RANDOM_STATE,
        multivariate=True,
        group=True
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=model_name
    )

    study.optimize(
        objective_func,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        n_jobs=1
    )
    return study


def trials_to_dataframe(study, model_name):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    rows = []
    best_so_far = np.inf

    for t in study.trials:
        if t.value is None:
            continue

        cv_rmse = float(t.value)
        best_so_far = min(best_so_far, cv_rmse)

        row = {
            "model": model_name,
            "trial": t.number,
            "cv_mean_RMSE": cv_rmse,
            "best_cv_mean_RMSE_so_far": best_so_far,
            "state": str(t.state),
        }
        row.update(t.params)
        rows.append(row)

    return pd.DataFrame(rows)


def get_adaptive_figsize(n_items, fig_type="parallel"):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    if fig_type == "parallel":
        width_cm = PARALLEL_FIG_WIDTH_CM
        height_cm = PARALLEL_FIG_HEIGHT_CM
    elif fig_type == "bar":
        width_cm = min(15.0, MAX_FIG_WIDTH_CM)
        height_cm = min(MAX_FIG_HEIGHT_CM, max(BASE_HEIGHT_CM, 0.52 * n_items + 3.5))
    elif fig_type == "dependence":
        
        width_cm = SHAP_DEPENDENCE_WIDTH_CM
        height_cm = SHAP_DEPENDENCE_HEIGHT_CM
    elif fig_type == "matrix":
        width_cm = min(18.0, MAX_FIG_WIDTH_CM)
        height_cm = min(18.0, MAX_FIG_HEIGHT_CM + 2.0)
    else:
        width_cm = MIN_FIG_WIDTH_CM
        height_cm = BASE_HEIGHT_CM

    return width_cm * CM_TO_INCH, height_cm * CM_TO_INCH


def plot_parallel_coordinates(df, model_name, save_path):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    plot_df = df.copy()
    if plot_df.empty:
        return

    exclude_cols = {"model", "trial", "state"}
    value_col = "cv_mean_RMSE"

    param_cols = [
        c for c in plot_df.columns
        if c not in exclude_cols and c != value_col and c != "best_cv_mean_RMSE_so_far"
    ]

    if len(param_cols) < 2:
        print(f"Skip parallel coordinate for {model_name}: fewer than 2 parameters.")
        return

    encoded = pd.DataFrame(index=plot_df.index)

    for col in param_cols:
        s = plot_df[col]

        
        if pd.api.types.is_bool_dtype(s):
            encoded[col] = s.astype(int).astype(float)
            continue

        
        if s.dtype == "object" or str(s.dtype).startswith("category"):
            s2 = s.astype(str).fillna("None")
            categories = sorted(s2.unique().tolist())
            mapping = {v: i for i, v in enumerate(categories)}
            encoded[col] = s2.map(mapping).astype(float)
            continue

        
        encoded[col] = pd.to_numeric(s, errors="coerce").astype(float)

    
    encoded[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce").astype(float)
    encoded = encoded.replace([np.inf, -np.inf], np.nan).dropna()

    if len(encoded) == 0:
        print(f"Skip parallel coordinate for {model_name}: no valid trials.")
        return

    
    norm_params = pd.DataFrame(index=encoded.index)
    for col in param_cols:
        col_min = encoded[col].min()
        col_max = encoded[col].max()

        if pd.isna(col_min) or pd.isna(col_max):
            norm_params[col] = 0.5
        elif np.isclose(float(col_min), float(col_max)):
            norm_params[col] = 0.5
        else:
            norm_params[col] = (encoded[col].astype(float) - float(col_min)) / (float(col_max) - float(col_min))

    values = encoded[value_col].astype(float).values
    norm = Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))
    cmap = COMMON_CMAP

    x_positions = np.arange(len(param_cols))

    fig, ax = plt.subplots(figsize=get_adaptive_figsize(len(param_cols), fig_type="parallel"))

    for i in range(len(norm_params)):
        y_vals = norm_params.iloc[i].values.astype(float)
        x_smooth, y_smooth = smooth_parallel_curve(x_positions, y_vals)
        ax.plot(
            x_smooth,
            y_smooth,
            color=cmap(norm(values[i])),
            linewidth=0.9,
            alpha=0.72
        )

    for x in x_positions:
        ax.axvline(x=x, color="gray", linewidth=0.6, alpha=0.5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_plot_label(c) for c in param_cols], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized value", labelpad=4)
    ax.set_xlabel("Hyperparameters", labelpad=4)
    ax.tick_params(axis="both", labelsize=11, pad=2)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.025, fraction=0.055)
    cbar.set_label("RMSE", fontsize=11, fontname="Times New Roman")
    cbar.ax.tick_params(labelsize=11)
    apply_times_new_roman(fig, font_size=11)

    fig.subplots_adjust(left=0.16, right=0.82, bottom=0.28, top=0.96)
    save_figure_dual(fig, save_path)
    plt.close(fig)



def plot_optimization_curve(trial_df, model_name, save_path):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    if trial_df is None or trial_df.empty:
        return
    if "cv_mean_RMSE" not in trial_df.columns:
        print(f"Skip optimization curve for {model_name}: cv_mean_RMSE not found.")
        return

    df_plot = trial_df.copy().sort_values("trial")
    df_plot["best_cv_mean_RMSE_so_far"] = df_plot["cv_mean_RMSE"].cummin()

    x = df_plot["trial"].values
    y = df_plot["cv_mean_RMSE"].values
    y_best = df_plot["best_cv_mean_RMSE_so_far"].values

    norm = Normalize(vmin=float(np.min(y)), vmax=float(np.max(y)))

    fig, ax = plt.subplots(figsize=(8.0 * CM_TO_INCH, 6.0 * CM_TO_INCH))

    sc = ax.scatter(
        x,
        y,
        c=y,
        cmap=COMMON_CMAP,
        norm=norm,
        s=28,
        alpha=0.78,
        edgecolors="black",
        linewidths=0.25,
        label="Current fitness"
    )

    ax.plot(
        x,
        y_best,
        color="black",
        linewidth=1.2,
        marker="o",
        markersize=2.8,
        label="Best fitness"
    )

    ax.set_xlabel("Iteration", fontsize=11, fontname="Times New Roman")
    ax.set_ylabel("RMSE", fontsize=11, fontname="Times New Roman")
    ax.tick_params(axis="both", labelsize=11, direction="in", length=3, width=0.8)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(11)

    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.35)

    leg = ax.legend(
        frameon=False,
        fontsize=11,
        loc="best",
        handlelength=1.8,
        borderaxespad=0.3
    )
    for text in leg.get_texts():
        text.set_fontname("Times New Roman")

    cbar = fig.colorbar(sc, ax=ax, pad=0.025, fraction=0.055)
    
    cbar.ax.tick_params(labelsize=11)
    for label in cbar.ax.get_yticklabels():
        label.set_fontname("Times New Roman")

    apply_times_new_roman(fig, font_size=11)
    fig.tight_layout()
    save_figure_dual(fig, save_path)
    plt.close(fig)



def compute_vaf(y_true, y_pred):
    """Variance Accounted For (VAF), reported in percentage."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    denom = np.var(y_true, ddof=1)
    if np.isclose(denom, 0.0):
        return np.nan
    return (1.0 - np.var(y_true - y_pred, ddof=1) / denom) * 100.0


def estimate_point_density(x, y, bins=40):
    """Estimate scatter-point density for coloring; prefer gaussian_kde, fallback to 2D histogram."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        return np.asarray(z, dtype=float)
    except Exception:
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
        xi = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
        yi = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
        z = hist[xi, yi]
        return np.asarray(z, dtype=float)


def plot_true_vs_pred_scatter(y_true, y_pred, model_name, dataset_name, save_path, target_label=TARGET_COL):
    """
    Plot actual-versus-predicted scatter for one model on one dataset, mimicking the provided sample style.
    Generates both PNG and SVG via save_figure_dual().
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return

    metrics = evaluate_regression(y_true, y_pred)
    vaf_value = compute_vaf(y_true, y_pred)

    density = estimate_point_density(y_true, y_pred)
    order = np.argsort(density)
    x = y_true[order]
    y = y_pred[order]
    z = density[order]

    data_min = float(min(np.min(x), np.min(y)))
    data_max = float(max(np.max(x), np.max(y)))
    span = data_max - data_min
    pad = 0.05 * span if span > 0 else 1.0
    axis_min = data_min - pad
    axis_max = data_max + pad

    line_x = np.array([axis_min, axis_max], dtype=float)
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        slope, intercept = 1.0, 0.0
    fit_y = slope * line_x + intercept

    fig, ax = plt.subplots(figsize=(9.0 * CM_TO_INCH, 6.8 * CM_TO_INCH))
    
    fig.patch.set_facecolor(PRED_SCATTER_FACE_COLOR)
    ax.set_facecolor(PRED_SCATTER_FACE_COLOR)

    sc = ax.scatter(
        x, y, c=z, cmap=COMMON_CMAP, s=10, alpha=0.95,
        edgecolors='black', linewidths=0.25
    )

    ax.plot(line_x, line_x, color='#2ca02c', linewidth=1.0, linestyle='-.')
    ax.plot(line_x, fit_y, color='black', linewidth=1.0, linestyle='-')
    ax.plot(line_x, 1.1 * line_x, color='#bf00bf', linewidth=0.9, linestyle='--')
    ax.plot(line_x, 0.9 * line_x, color='#bf00bf', linewidth=0.9, linestyle='--')

    ax.text(0.73, 0.89, 'y=1.1x', color='#bf00bf', fontsize=10, fontname='Times New Roman', transform=ax.transAxes)
    ax.text(0.77, 0.53, 'y=0.9x', color='#bf00bf', fontsize=10, fontname='Times New Roman', transform=ax.transAxes)

    set_name = 'Training' if str(dataset_name).lower().startswith('train') else 'Testing'
    text_block = (
        f'{set_name} set\n'
        f'R$^2$: {metrics["R2"]:.4f}\n'
        f'MAE: {metrics["MAE"]:.4f}\n'
        f'RMSE: {metrics["RMSE"]:.4f}\n'
        f'VAF: {vaf_value:.4f}'
    )
    ax.text(0.06, 0.96, text_block, transform=ax.transAxes, va='top', ha='left', fontsize=10)
    ax.text(0.95, 0.08, model_name, transform=ax.transAxes, ha='right', va='center', fontsize=10)

    axis_label = format_plot_label(target_label)
    ax.set_xlabel(f'Actual {axis_label}', fontsize=11, fontname='Times New Roman')
    ax.set_ylabel(f'Predicted {axis_label}', fontsize=11, fontname='Times New Roman')
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.tick_params(axis='both', labelsize=11, direction='in', length=3, width=0.8)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.055)
    cbar.ax.tick_params(labelsize=10)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    apply_times_new_roman(fig, font_size=11)
    fig.tight_layout()
    save_figure_dual(fig, save_path)
    plt.close(fig)


def fit_and_evaluate_best_model(model, X_train, y_train, X_test, y_test, scale_numeric=False):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    y_train_log = np.log1p(y_train)

    preprocessor, _, _ = make_preprocessor(X_train, scale_numeric=scale_numeric)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train_log)

    train_pred_log = pipe.predict(X_train)
    test_pred_log = pipe.predict(X_test)

    train_pred = safe_expm1(train_pred_log)
    test_pred = safe_expm1(test_pred_log)

    train_metrics = evaluate_regression(y_train, train_pred)
    test_metrics = evaluate_regression(y_test, test_pred)

    return pipe, train_pred, test_pred, train_metrics, test_metrics


# =========================

# =========================

df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)


required_cols = [TARGET_COL, REGION_COL]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Cannot find required column: {col}")


df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)


province_series = df[PROVINCE_COL].copy() if PROVINCE_COL in df.columns else pd.Series(["Unknown"] * len(df))


drop_cols = [TARGET_COL]
if PROVINCE_COL in df.columns:
    drop_cols.append(PROVINCE_COL)

X = df.drop(columns=drop_cols)

actual_drop_features = [c for c in DROP_FEATURE_COLS if c in X.columns]
if actual_drop_features:
    X = X.drop(columns=actual_drop_features)
    print("Dropped input features:", actual_drop_features)
y = df[TARGET_COL].astype(float)


if REGION_COL in X.columns:
    X[REGION_COL] = X[REGION_COL].astype(str)


X_train, X_test, y_train, y_test, prov_train, prov_test = train_test_split(
    X,
    y,
    province_series,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

y_train_log = np.log1p(y_train)

print("Data shape:", df.shape)
print("Feature shape:", X.shape)
print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])
print("Input features do NOT include Province.")
print("Input features include Region:", REGION_COL in X.columns)


feature_meta_df = save_feature_metadata_table(X.columns.tolist(), OUT_DIR)


# =========================

# =========================

def objective_adaboost(trial):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    from sklearn.tree import DecisionTreeRegressor

    max_depth = trial.suggest_int("max_depth", 2, 8)
    base_estimator = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=RANDOM_STATE
    )

    ada_params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 700),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        loss=trial.suggest_categorical("loss", ["linear", "square", "exponential"]),
        random_state=RANDOM_STATE,
    )

    
    
    try:
        model = AdaBoostRegressor(estimator=base_estimator, **ada_params)
    except TypeError:
        model = AdaBoostRegressor(base_estimator=base_estimator, **ada_params)

    return objective_cv_score(model, X_train, y_train_log, scale_numeric=False)

def objective_rf(trial):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 700),
        max_depth=trial.suggest_int("max_depth", 2, 20),
        min_samples_split=2,
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
        max_features=trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.7, 1.0]),
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return objective_cv_score(model, X_train, y_train_log, scale_numeric=False)

def objective_gbdt(trial):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    model = GradientBoostingRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 700),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=RANDOM_STATE,
    )
    return objective_cv_score(model, X_train, y_train_log, scale_numeric=False)

def objective_xgb(trial):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    if XGBRegressor is None:
        raise optuna.exceptions.TrialPruned("xgboost is not installed.")

    model = XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 700),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=1.0,
        min_child_weight=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    return objective_cv_score(model, X_train, y_train_log, scale_numeric=False)

def objective_lgbm(trial):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    if LGBMRegressor is None:
        raise optuna.exceptions.TrialPruned("lightgbm is not installed.")

    model = LGBMRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 700),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        num_leaves=trial.suggest_int("num_leaves", 8, 64),
        min_child_samples=20,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    return objective_cv_score(model, X_train, y_train_log, scale_numeric=False)

def objective_catboost(trial):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    if CatBoostRegressor is None:
        raise optuna.exceptions.TrialPruned("catboost is not installed.")

    model = CatBoostRegressor(
        iterations=trial.suggest_int("iterations", 100, 700),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        depth=trial.suggest_int("depth", 2, 8),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.1, 20.0, log=True),
        bagging_temperature=1.0,
        random_strength=1.0,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    return objective_cv_score(model, X_train, y_train_log, scale_numeric=False)

OBJECTIVES = {
    "AdaBoost": objective_adaboost,
    "XGBoost": objective_xgb,
    "RF": objective_rf,
    "GBDT": objective_gbdt,
    "CatBoost": objective_catboost,
    "LGBM": objective_lgbm,
}


def build_model_from_params(model_name, params):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    params = params.copy()

    if model_name == "AdaBoost":
        from sklearn.tree import DecisionTreeRegressor

        max_depth = params.pop("max_depth")
        base_estimator = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=RANDOM_STATE
        )

        try:
            return AdaBoostRegressor(
                estimator=base_estimator,
                **params,
                random_state=RANDOM_STATE,
            ), False
        except TypeError:
            return AdaBoostRegressor(
                base_estimator=base_estimator,
                **params,
                random_state=RANDOM_STATE,
            ), False

    if model_name == "RF":
        return RandomForestRegressor(
            **params,
            min_samples_split=2,
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ), False

    if model_name == "GBDT":
        return GradientBoostingRegressor(
            **params,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            random_state=RANDOM_STATE,
        ), False

    if model_name == "XGBoost":
        return XGBRegressor(
            **params,
            colsample_bytree=1.0,
            min_child_weight=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ), False

    if model_name == "LGBM":
        return LGBMRegressor(
            **params,
            min_child_samples=20,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ), False

    if model_name == "CatBoost":
        return CatBoostRegressor(
            **params,
            bagging_temperature=1.0,
            random_strength=1.0,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        ), False

    raise ValueError(f"Unknown model: {model_name}")

# =========================

# =========================

all_trial_dfs = []
summary_rows = []
failed_rows = []
fitted_models = {}
all_model_prediction_dfs = []  

for model_name, objective_func in OBJECTIVES.items():
    print("\n" + "=" * 80)
    print(f"Optimizing {model_name}")
    print("=" * 80)

    try:
        study = run_optuna(model_name, objective_func)
    except Exception as e:
        err_msg = repr(e)
        print(f"Optimization failed for {model_name}: {err_msg}")
        failed_rows.append({"model": model_name, "error": err_msg})
        continue

    trial_df = trials_to_dataframe(study, model_name)
    all_trial_dfs.append(trial_df)

    
    svg_path = FIG_DIR / f"{model_name}_parallel_coordinates.svg"
    plot_parallel_coordinates(trial_df, model_name, svg_path)

    
    curve_path = FIG_DIR / f"{model_name}_optimization_curve.svg"
    plot_optimization_curve(trial_df, model_name, curve_path)

    
    best_params = study.best_params
    best_model, scale_numeric = build_model_from_params(model_name, best_params)

    fitted_pipe, train_pred, test_pred, train_metrics, test_metrics = fit_and_evaluate_best_model(
        best_model,
        X_train,
        y_train,
        X_test,
        y_test,
        scale_numeric=scale_numeric
    )

    fitted_models[model_name] = {
        "pipeline": fitted_pipe,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "pred": test_pred,  
        "best_params": best_params,
        "scale_numeric": scale_numeric,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    
    
    model_train_pred_df = X_train.copy()
    model_train_pred_df.insert(0, "model", model_name)
    model_train_pred_df.insert(1, "Dataset", "Train")
    model_train_pred_df.insert(2, "Sample_Index", X_train.index.values)
    model_train_pred_df[PROVINCE_COL] = prov_train.values
    model_train_pred_df["y_true"] = y_train.values
    model_train_pred_df["y_pred"] = train_pred
    model_train_pred_df["error"] = model_train_pred_df["y_pred"] - model_train_pred_df["y_true"]
    model_train_pred_df["abs_error"] = np.abs(model_train_pred_df["error"])

    model_test_pred_df = X_test.copy()
    model_test_pred_df.insert(0, "model", model_name)
    model_test_pred_df.insert(1, "Dataset", "Test")
    model_test_pred_df.insert(2, "Sample_Index", X_test.index.values)
    model_test_pred_df[PROVINCE_COL] = prov_test.values
    model_test_pred_df["y_true"] = y_test.values
    model_test_pred_df["y_pred"] = test_pred
    model_test_pred_df["error"] = model_test_pred_df["y_pred"] - model_test_pred_df["y_true"]
    model_test_pred_df["abs_error"] = np.abs(model_test_pred_df["error"])

    all_model_prediction_dfs.extend([model_train_pred_df, model_test_pred_df])

    
    plot_true_vs_pred_scatter(
        y_train.values,
        train_pred,
        model_name=model_name,
        dataset_name="Train",
        save_path=PRED_SCATTER_DIR / f"{model_name}_Train_true_vs_pred.svg",
        target_label=TARGET_COL
    )
    plot_true_vs_pred_scatter(
        y_test.values,
        test_pred,
        model_name=model_name,
        dataset_name="Test",
        save_path=PRED_SCATTER_DIR / f"{model_name}_Test_true_vs_pred.svg",
        target_label=TARGET_COL
    )

    row = {
        "model": model_name,
        "best_cv_mean_RMSE": study.best_value,
        "best_params": json.dumps(best_params, ensure_ascii=False),
    }
    for metric_name, metric_value in train_metrics.items():
        row[f"train_{metric_name}"] = metric_value
    for metric_name, metric_value in test_metrics.items():
        row[f"test_{metric_name}"] = metric_value
    summary_rows.append(row)

    print(f"{model_name} train metrics:", train_metrics)
    print(f"{model_name} test metrics:", test_metrics)
    print("Best params:", best_params)


summary_df = pd.DataFrame(summary_rows)
if not summary_df.empty and "test_R2" in summary_df.columns:
    summary_df = summary_df.sort_values("test_R2", ascending=False)
else:
    print("\nNo model was successfully optimized. Please check Failed_Models sheet and console errors.")
    base_cols = ["model", "best_cv_mean_RMSE", "best_params"]
    metric_cols = ["R2", "RMSE", "MSE", "MAE", "MedAE", "MAPE_percent", "sMAPE_percent", "Explained_variance", "RPD", "NSE"]
    summary_df = pd.DataFrame(columns=base_cols + [f"train_{m}" for m in metric_cols] + [f"test_{m}" for m in metric_cols])


performance_long_rows = []
if not summary_df.empty:
    metric_cols = ["R2", "RMSE", "MSE", "MAE", "MedAE", "MAPE_percent", "sMAPE_percent", "Explained_variance", "RPD", "NSE"]
    for _, r in summary_df.iterrows():
        for dataset_name, prefix in [("Train", "train_"), ("Test", "test_")]:
            item = {
                "model": r.get("model"),
                "Dataset": dataset_name,
                "best_cv_mean_RMSE": r.get("best_cv_mean_RMSE"),
            }
            for m in metric_cols:
                item[m] = r.get(prefix + m, np.nan)
            performance_long_rows.append(item)
performance_long_df = pd.DataFrame(performance_long_rows)

failed_df = pd.DataFrame(failed_rows)

if len(all_trial_dfs) > 0:
    trials_df = pd.concat(all_trial_dfs, axis=0, ignore_index=True)
else:
    trials_df = pd.DataFrame()


result_excel = OUT_DIR / "Optuna_optimization_and_model_performance.xlsx"
with pd.ExcelWriter(result_excel, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Model_Performance_Wide", index=False)
    performance_long_df.to_excel(writer, sheet_name="Model_Performance_Long", index=False)
    trials_df.to_excel(writer, sheet_name="All_Optuna_Trials", index=False)
    failed_df.to_excel(writer, sheet_name="Failed_Models", index=False)




all_models_pred_path = OUT_DIR / "All_models_train_test_true_pred_values.xlsx"
if len(all_model_prediction_dfs) > 0:
    all_models_pred_df = pd.concat(all_model_prediction_dfs, axis=0, ignore_index=True)
    all_models_train_pred_df = all_models_pred_df[all_models_pred_df["Dataset"] == "Train"].copy()
    all_models_test_pred_df = all_models_pred_df[all_models_pred_df["Dataset"] == "Test"].copy()
else:
    all_models_pred_df = pd.DataFrame(columns=["model", "Dataset", "Sample_Index", PROVINCE_COL, "y_true", "y_pred", "error", "abs_error"])
    all_models_train_pred_df = all_models_pred_df.copy()
    all_models_test_pred_df = all_models_pred_df.copy()

with pd.ExcelWriter(all_models_pred_path, engine="openpyxl") as writer:
    all_models_pred_df.to_excel(writer, sheet_name="All_Model_Predictions", index=False)
    all_models_train_pred_df.to_excel(writer, sheet_name="Train_Predictions", index=False)
    all_models_test_pred_df.to_excel(writer, sheet_name="Test_Predictions", index=False)

    for m in sorted(all_models_pred_df["model"].dropna().unique()):
        sub_train = all_models_pred_df[(all_models_pred_df["model"] == m) & (all_models_pred_df["Dataset"] == "Train")].copy()
        sub_test = all_models_pred_df[(all_models_pred_df["model"] == m) & (all_models_pred_df["Dataset"] == "Test")].copy()
        safe_m = str(m).replace("/", "_").replace("\\", "_")[:20]
        sub_train.to_excel(writer, sheet_name=f"{safe_m}_Train"[:31], index=False)
        sub_test.to_excel(writer, sheet_name=f"{safe_m}_Test"[:31], index=False)

print("Saved all models train/test true-pred values to:", all_models_pred_path)



optuna_process_excel = OUT_DIR / "Optuna_iteration_hyperparameters_and_fitness.xlsx"
with pd.ExcelWriter(optuna_process_excel, engine="openpyxl") as writer:
    if not trials_df.empty:
        trials_df.to_excel(writer, sheet_name="All_Models_Iterations", index=False)
        for m in sorted(trials_df["model"].dropna().unique()):
            sub = trials_df[trials_df["model"] == m].copy()
            safe_sheet = str(m)[:31]
            sub.to_excel(writer, sheet_name=safe_sheet, index=False)
    else:
        pd.DataFrame(columns=["model", "trial", "cv_mean_RMSE", "best_cv_mean_RMSE_so_far"]).to_excel(
            writer, sheet_name="All_Models_Iterations", index=False
        )

print("\nSaved optimization results to:", result_excel)
print("Saved Optuna iteration hyperparameters and fitness to:", optuna_process_excel)
print("\nModel performance summary:")
print(summary_df)


search_space_rows = [
    {"Model": "AdaBoost", "Hyperparameter": "n_estimators", "Search space": "100–700", "Role": "Number of weak learners"},
    {"Model": "AdaBoost", "Hyperparameter": "learning_rate", "Search space": "0.01–0.15 (log)", "Role": "Boosting learning rate"},
    {"Model": "AdaBoost", "Hyperparameter": "loss", "Search space": "linear, square, exponential", "Role": "Loss function"},
    {"Model": "AdaBoost", "Hyperparameter": "max_depth", "Search space": "2–8", "Role": "Tree depth"},

    {"Model": "XGBoost", "Hyperparameter": "n_estimators", "Search space": "100–700", "Role": "Number of trees"},
    {"Model": "XGBoost", "Hyperparameter": "learning_rate", "Search space": "0.01–0.15 (log)", "Role": "Boosting learning rate"},
    {"Model": "XGBoost", "Hyperparameter": "max_depth", "Search space": "2–8", "Role": "Tree depth"},
    {"Model": "XGBoost", "Hyperparameter": "subsample", "Search space": "0.6–1.0", "Role": "Row sampling ratio"},

    {"Model": "RF", "Hyperparameter": "n_estimators", "Search space": "100–700", "Role": "Number of trees"},
    {"Model": "RF", "Hyperparameter": "max_depth", "Search space": "2–20", "Role": "Tree depth"},
    {"Model": "RF", "Hyperparameter": "min_samples_leaf", "Search space": "1–8", "Role": "Minimum samples per leaf"},
    {"Model": "RF", "Hyperparameter": "max_features", "Search space": "sqrt, 0.5, 0.7, 1.0", "Role": "Feature sampling ratio"},

    {"Model": "GBDT", "Hyperparameter": "n_estimators", "Search space": "100–700", "Role": "Number of trees"},
    {"Model": "GBDT", "Hyperparameter": "learning_rate", "Search space": "0.01–0.15 (log)", "Role": "Boosting learning rate"},
    {"Model": "GBDT", "Hyperparameter": "max_depth", "Search space": "2–8", "Role": "Tree depth"},
    {"Model": "GBDT", "Hyperparameter": "subsample", "Search space": "0.6–1.0", "Role": "Row sampling ratio"},

    {"Model": "CatBoost", "Hyperparameter": "iterations", "Search space": "100–700", "Role": "Number of trees"},
    {"Model": "CatBoost", "Hyperparameter": "learning_rate", "Search space": "0.01–0.15 (log)", "Role": "Boosting learning rate"},
    {"Model": "CatBoost", "Hyperparameter": "depth", "Search space": "2–8", "Role": "Tree depth"},
    {"Model": "CatBoost", "Hyperparameter": "l2_leaf_reg", "Search space": "0.1–20 (log)", "Role": "L2 regularization"},

    {"Model": "LGBM", "Hyperparameter": "n_estimators", "Search space": "100–700", "Role": "Number of trees"},
    {"Model": "LGBM", "Hyperparameter": "learning_rate", "Search space": "0.01–0.15 (log)", "Role": "Boosting learning rate"},
    {"Model": "LGBM", "Hyperparameter": "max_depth", "Search space": "2–8", "Role": "Tree depth"},
    {"Model": "LGBM", "Hyperparameter": "num_leaves", "Search space": "8–64", "Role": "Leaf-wise tree complexity"},
]
search_space_df = pd.DataFrame(search_space_rows)
search_space_path = OUT_DIR / "Hyperparameter_search_space_table.xlsx"
search_space_df.to_excel(search_space_path, index=False)
print("Saved hyperparameter search-space table to:", search_space_path)
print("\\nHyperparameter search-space table:")
print(search_space_df)



# =========================

# =========================

if summary_df.empty or summary_df["test_R2"].isna().all():
    raise RuntimeError(
        "No model was successfully optimized. "
        "Open Optuna_Model_Results/Optuna_optimization_and_model_performance.xlsx "
        "and check the Failed_Models sheet."
    )

best_model_name = summary_df.iloc[0]["model"]
best_info = fitted_models[best_model_name]
best_pipe = best_info["pipeline"]
best_pred = best_info["test_pred"]

print("\nBest model:", best_model_name)



train_pred = best_info["train_pred"]
test_pred = best_pred

train_pred_df = X_train.copy()
train_pred_df[PROVINCE_COL] = prov_train.values
train_pred_df["Dataset"] = "Train"
train_pred_df["y_true"] = y_train.values
train_pred_df["y_pred"] = train_pred
train_pred_df["error"] = train_pred_df["y_pred"] - train_pred_df["y_true"]
train_pred_df["abs_error"] = np.abs(train_pred_df["error"])

test_pred_df = X_test.copy()
test_pred_df[PROVINCE_COL] = prov_test.values
test_pred_df["Dataset"] = "Test"
test_pred_df["y_true"] = y_test.values
test_pred_df["y_pred"] = test_pred
test_pred_df["error"] = test_pred_df["y_pred"] - test_pred_df["y_true"]
test_pred_df["abs_error"] = np.abs(test_pred_df["error"])

pred_all_df = pd.concat([train_pred_df, test_pred_df], axis=0, ignore_index=True)

pred_path = OUT_DIR / "Best_model_train_test_true_pred_values.xlsx"
with pd.ExcelWriter(pred_path, engine="openpyxl") as writer:
    train_pred_df.to_excel(writer, sheet_name="Train_true_pred", index=False)
    test_pred_df.to_excel(writer, sheet_name="Test_true_pred", index=False)
    pred_all_df.to_excel(writer, sheet_name="All_true_pred", index=False)

print("Saved train/test true-pred values to:", pred_path)



def save_shap_replot_data(
    shap_values,
    X_exp_processed,
    feature_names,
    shap_plot_values,
    X_plot_abbr,
    abbr_names,
    mean_abs_plot,
    model_name,
    out_dir
):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shap_arr = np.asarray(shap_values)
    shap_plot_arr = np.asarray(shap_plot_values)

    raw_shap_df = pd.DataFrame(shap_arr, columns=feature_names, index=X_exp_processed.index)
    raw_x_df = X_exp_processed.copy()
    raw_x_df.columns = feature_names

    plot_shap_df = pd.DataFrame(shap_plot_arr, columns=abbr_names, index=X_plot_abbr.index)
    plot_x_df = X_plot_abbr.copy()

    mapping_df = pd.DataFrame({
        "plot_feature_name": abbr_names,
        "note": "These columns correspond to the merged/abbreviated SHAP plotting matrix. Region one-hot variables are merged when applicable."
    })

    replot_excel = out_dir / f"{model_name}_SHAP_replot_data.xlsx"
    with pd.ExcelWriter(replot_excel, engine="openpyxl") as writer:
        raw_shap_df.to_excel(writer, sheet_name="raw_SHAP_values", index=True)
        raw_x_df.to_excel(writer, sheet_name="raw_feature_values", index=True)
        plot_shap_df.to_excel(writer, sheet_name="plot_SHAP_values", index=True)
        plot_x_df.to_excel(writer, sheet_name="plot_feature_values", index=True)
        mean_abs_plot.to_excel(writer, sheet_name="plot_mean_abs_SHAP", index=False)
        mapping_df.to_excel(writer, sheet_name="plot_feature_mapping", index=False)

    
    plot_shap_csv = out_dir / f"{model_name}_plot_SHAP_values_for_replot.csv"
    plot_x_csv = out_dir / f"{model_name}_plot_feature_values_for_replot.csv"
    mean_abs_csv = out_dir / f"{model_name}_plot_mean_abs_SHAP_for_replot.csv"
    plot_shap_df.to_csv(plot_shap_csv, encoding="utf-8-sig")
    plot_x_df.to_csv(plot_x_csv, encoding="utf-8-sig")
    mean_abs_plot.to_csv(mean_abs_csv, index=False, encoding="utf-8-sig")

    print("Saved SHAP replot data to:", replot_excel)
    print("Saved SHAP replot CSV files to:", out_dir)


def plot_shap_combined_summary(
    shap_plot_values,
    X_plot_abbr,
    mean_abs_plot,
    model_name,
    out_dir,
    max_display=20
):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shap_arr = np.asarray(shap_plot_values)
    features_all = list(X_plot_abbr.columns)

    top_features = mean_abs_plot["feature"].head(min(max_display, len(mean_abs_plot))).tolist()
    plot_features = top_features[::-1]  
    y_positions = np.arange(len(plot_features))

    fig_height_cm = min(MAX_FIG_HEIGHT_CM, max(BASE_HEIGHT_CM, 0.52 * len(plot_features) + 3.5))
    fig, ax = plt.subplots(figsize=(15.0 * CM_TO_INCH, fig_height_cm * CM_TO_INCH))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    
    bar_values = []
    for feat in plot_features:
        val = mean_abs_plot.loc[mean_abs_plot["feature"] == feat, "mean_abs_SHAP"].values
        bar_values.append(float(val[0]) if len(val) > 0 else 0.0)

    ax_bar = ax.twiny()
    ax_bar.barh(
        y_positions,
        bar_values,
        height=0.72,
        color=SHAP_COMBINED_BAR_COLOR,
        alpha=SHAP_COMBINED_BAR_ALPHA,
        edgecolor="none",
        zorder=0
    )
    ax_bar.set_xlabel("Mean |SHAP value|", fontsize=11, fontname="Times New Roman")
    ax_bar.tick_params(axis="x", labelsize=11, direction="in", length=3, width=0.8)
    ax_bar.grid(False)
    ax_bar.set_zorder(0)

    
    rng = np.random.default_rng(RANDOM_STATE)
    for yi, feat in enumerate(plot_features):
        if feat not in features_all:
            continue

        idx = features_all.index(feat)
        shap_vals = shap_arr[:, idx].astype(float)
        feat_vals = pd.to_numeric(X_plot_abbr[feat], errors="coerce").astype(float).values

        valid = ~(np.isnan(shap_vals) | np.isnan(feat_vals))
        if valid.sum() == 0:
            continue

        shap_vals = shap_vals[valid]
        feat_vals = feat_vals[valid]

        
        f_min, f_max = np.nanmin(feat_vals), np.nanmax(feat_vals)
        if np.isclose(f_min, f_max):
            color_vals = np.full_like(feat_vals, 0.5, dtype=float)
        else:
            color_vals = (feat_vals - f_min) / (f_max - f_min)

        
        jitter = rng.normal(0, 0.075, size=len(shap_vals))
        jitter = np.clip(jitter, -0.22, 0.22)

        ax.scatter(
            shap_vals,
            np.full_like(shap_vals, yi, dtype=float) + jitter,
            c=color_vals,
            cmap=COMMON_CMAP,
            vmin=0.0,
            vmax=1.0,
            s=18,
            alpha=0.82,
            edgecolors="none",
            zorder=3
        )

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_features, fontsize=11, fontname="Times New Roman")
    ax.set_xlabel("SHAP value", fontsize=11, fontname="Times New Roman")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=11, direction="in", length=3, width=0.8)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.45, alpha=0.30)
    ax.set_zorder(1)
    ax.patch.set_alpha(0.0)

    
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=COMMON_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.025, fraction=0.045)
    cbar.set_label("Feature value", fontsize=11, fontname="Times New Roman")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=11)

    apply_times_new_roman(fig, font_size=11)
    fig.tight_layout()
    combined_path = out_dir / f"{model_name}_SHAP_summary_combined_dot_bar_region_merged.svg"
    save_figure_dual(fig, combined_path)
    plt.close(fig)
    print("Saved SHAP combined summary figure to:", combined_path)


def save_shap_svg_plots(shap_values, X_exp_processed, feature_names, mean_abs_shap, model_name, out_dir):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    if shap is None:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11
    plt.rcParams["svg.fonttype"] = "path"

    
    shap_plot_values, X_plot, plot_feature_names = merge_region_shap_values(
        shap_values, X_exp_processed, feature_names
    )

    
    abbr_names = get_abbr_columns(plot_feature_names)
    X_plot_abbr = X_plot.copy()
    X_plot_abbr.columns = abbr_names

    mean_abs_plot = (
        pd.DataFrame({
            "feature": abbr_names,
            "mean_abs_SHAP": np.abs(shap_plot_values).mean(axis=0)
        })
        .sort_values("mean_abs_SHAP", ascending=False)
        .reset_index(drop=True)
    )

    
    save_shap_replot_data(
        shap_values=shap_values,
        X_exp_processed=X_exp_processed,
        feature_names=feature_names,
        shap_plot_values=shap_plot_values,
        X_plot_abbr=X_plot_abbr,
        abbr_names=abbr_names,
        mean_abs_plot=mean_abs_plot,
        model_name=model_name,
        out_dir=out_dir
    )

    n_features = min(20, len(abbr_names))

    # 1. SHAP summary dot plot
    plt.figure(figsize=get_adaptive_figsize(n_features, fig_type="bar"))
    shap.summary_plot(
        shap_plot_values,
        X_plot_abbr,
        feature_names=abbr_names,
        max_display=n_features,
        show=False,
        plot_size=None,
        cmap=COMMON_CMAP
    )
    fig = plt.gcf()
    for ax in fig.axes:
        ax.tick_params(labelsize=11)
        ax.set_xlabel(ax.get_xlabel(), fontsize=11, fontname="Times New Roman")
        ax.set_ylabel(ax.get_ylabel(), fontsize=11, fontname="Times New Roman")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(11)
    apply_times_new_roman(fig, font_size=11)
    fig.tight_layout()
    summary_svg = out_dir / f"{model_name}_SHAP_summary_dot_region_merged.svg"
    save_figure_dual(fig, summary_svg)
    plt.close(fig)
    print("Saved SHAP summary dot figure to:", summary_svg)

    # 2. Custom SHAP bar plot
    top_n = min(20, len(mean_abs_plot))
    tmp = mean_abs_plot.head(top_n).iloc[::-1].copy()
    bar_vals = tmp["mean_abs_SHAP"].values.astype(float)
    bar_norm = Normalize(vmin=float(np.min(bar_vals)), vmax=float(np.max(bar_vals)))
    bar_colors = [COMMON_CMAP(bar_norm(v)) for v in bar_vals]

    fig, ax = plt.subplots(figsize=get_adaptive_figsize(top_n, fig_type="bar"))
    ax.barh(tmp["feature"], tmp["mean_abs_SHAP"], color=bar_colors, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11, fontname="Times New Roman")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=11)
    sm = ScalarMappable(norm=bar_norm, cmap=COMMON_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.025, fraction=0.055)
    cbar.set_label("Mean |SHAP value|", fontsize=11, fontname="Times New Roman")
    cbar.ax.tick_params(labelsize=11)
    apply_times_new_roman(fig, font_size=11)
    fig.tight_layout()
    bar_svg = out_dir / f"{model_name}_SHAP_summary_bar_region_merged.svg"
    save_figure_dual(fig, bar_svg)
    plt.close(fig)
    print("Saved SHAP summary bar figure to:", bar_svg)

    # 3. SHAP combined dot + transparent bar summary plot
    plot_shap_combined_summary(
        shap_plot_values=shap_plot_values,
        X_plot_abbr=X_plot_abbr,
        mean_abs_plot=mean_abs_plot,
        model_name=model_name,
        out_dir=out_dir,
        max_display=20
    )

    # 4. Top feature dependence/scatter plots with colorbar
    top_features = mean_abs_plot["feature"].head(min(6, len(mean_abs_plot))).tolist()
    for feature in top_features:
        if feature not in X_plot_abbr.columns:
            continue

        feature_vals = pd.to_numeric(X_plot_abbr[feature], errors="coerce")
        shap_idx = list(X_plot_abbr.columns).index(feature)
        shap_feat_vals = np.asarray(shap_plot_values)[:, shap_idx]

        mask = ~(pd.isna(feature_vals) | pd.isna(shap_feat_vals))
        if mask.sum() < 5:
            continue

        xv = feature_vals[mask].astype(float).values
        yv = np.asarray(shap_feat_vals)[mask].astype(float)
        norm = Normalize(vmin=float(np.min(xv)), vmax=float(np.max(xv)))

        fig, ax = plt.subplots(figsize=get_adaptive_figsize(1, fig_type="dependence"))
        sc = ax.scatter(
            xv, yv,
            c=xv,
            cmap=COMMON_CMAP,
            norm=norm,
            s=18,
            alpha=0.75,
            edgecolors="none"
        )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(feature, fontsize=11, fontname="Times New Roman")
        ax.set_ylabel(f"SHAP value for {feature}", fontsize=11, fontname="Times New Roman")
        ax.tick_params(axis="both", labelsize=11)

        cbar = fig.colorbar(sc, ax=ax, pad=0.03, fraction=0.05)
        cbar.set_label(f"{feature} value", fontsize=11, fontname="Times New Roman")
        cbar.ax.tick_params(labelsize=11)

        apply_times_new_roman(fig, font_size=11)
        fig.tight_layout()
        safe_feature = "".join([ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in feature])[:80]
        dep_svg = out_dir / f"{model_name}_SHAP_dependence_{safe_feature}.svg"
        save_figure_dual(fig, dep_svg)
        plt.close(fig)
        print("Saved SHAP dependence figure to:", dep_svg)



def save_region_shap_analysis(
    shap_values, X_exp_processed, feature_names, X_original_subset, province_subset, model_name, region_col, base_out_dir
):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    base_out_dir = Path(base_out_dir)
    if region_col not in X_original_subset.columns:
        print(f"Column {region_col} not found, skip region-level SHAP analysis.")
        return

    region_root = base_out_dir / "by_region"
    region_root.mkdir(parents=True, exist_ok=True)

    region_series = X_original_subset[region_col].astype(str)
    unique_regions = sorted(region_series.dropna().unique().tolist())

    shap_array = np.asarray(shap_values)

    for region in unique_regions:
        idx = region_series[region_series == region].index
        if len(idx) < 5:
            continue

        safe_region = "".join([ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in str(region)])[:80]
        region_dir = region_root / safe_region
        region_dir.mkdir(parents=True, exist_ok=True)

        X_reg_proc = X_exp_processed.loc[idx].copy()
        shap_reg = shap_array[[X_exp_processed.index.get_loc(i) for i in idx], :]

        region_sample_df = pd.DataFrame(shap_reg, columns=feature_names, index=idx)
        region_sample_df["Province"] = province_subset.loc[idx].values
        region_sample_path = region_dir / f"{model_name}_{safe_region}_SHAP_values.xlsx"
        region_sample_df.to_excel(region_sample_path, index=True)

        mean_abs = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_SHAP": np.abs(shap_reg).mean(axis=0)
        }).sort_values("mean_abs_SHAP", ascending=False).reset_index(drop=True)
        mean_abs_path = region_dir / f"{model_name}_{safe_region}_global_SHAP_importance.xlsx"
        mean_abs.to_excel(mean_abs_path, index=False)

        save_shap_svg_plots(
            shap_values=shap_reg,
            X_exp_processed=X_reg_proc,
            feature_names=feature_names,
            mean_abs_shap=mean_abs,
            model_name=f"{model_name}_{safe_region}",
            out_dir=region_dir
        )



def save_pdp_matrix(
    best_pipe,
    X_train,
    mean_abs_shap,
    out_dir,
    model_name,
    top_k=None,
    grid_resolution=20,
    max_features_per_matrix=None
):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = best_pipe.named_steps["preprocess"]
    model = best_pipe.named_steps["model"]

    X_train_proc = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out().tolist()
    X_train_proc_df = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)

    feature_to_idx = {f: i for i, f in enumerate(feature_names)}

    ordered_features = mean_abs_shap["feature"].tolist()

    candidate_features = []
    for feat in ordered_features:
        if feat not in X_train_proc_df.columns:
            continue

        x = pd.to_numeric(X_train_proc_df[feat], errors="coerce")
        nunique = x.nunique(dropna=True)

        
        unique_vals = set(pd.Series(x.dropna().unique()).round(10).tolist())
        is_binary = unique_vals.issubset({0, 1, 0.0, 1.0})

        if nunique >= 8 and not is_binary:
            candidate_features.append(feat)

    if len(candidate_features) < 2:
        print("Not enough continuous features for 2D PDP matrix.")
        return

    
    pdp_feature_df = pd.DataFrame({
        "feature_full": candidate_features,
        "feature_abbr": [format_plot_label(abbr_feature_name(f) if "abbr_feature_name" in globals() else f) for f in candidate_features],
        "rank_by_SHAP": list(range(1, len(candidate_features) + 1))
    })
    pdp_feature_path = out_dir / f"{model_name}_2D_PDP_candidate_features_all_continuous.xlsx"
    pdp_feature_df.to_excel(pdp_feature_path, index=False)
    print("Saved all 2D PDP candidate features to:", pdp_feature_path)

    def _plot_one_pdp_matrix(features_subset, suffix, figure_role):
        n = len(features_subset)
        if n < 2:
            print(f"Skip 2D PDP matrix {suffix}: fewer than 2 features.")
            return

        print(f"Generating 2D PDP matrix [{figure_role}] with {n} features: {features_subset}")

        pd_results = {}
        zmins, zmaxs = [], []

        for i in range(n):
            for j in range(i):
                f1 = features_subset[j]
                f2 = features_subset[i]

                idx1 = feature_to_idx[f1]
                idx2 = feature_to_idx[f2]

                try:
                    res = partial_dependence(
                        model,
                        X_train_proc_df,
                        features=[(idx1, idx2)],
                        grid_resolution=grid_resolution,
                        kind="average"
                    )
                except Exception as e:
                    print(f"Skip 2D PDP for ({f1}, {f2}) because: {repr(e)}")
                    continue

                avg = res["average"][0]
                if "grid_values" in res:
                    grids = res["grid_values"]
                else:
                    grids = res["values"]

                xgrid = np.array(grids[0], dtype=float)
                ygrid = np.array(grids[1], dtype=float)
                z = np.array(avg, dtype=float)

                if z.shape == (len(ygrid), len(xgrid)):
                    z_plot = z
                elif z.shape == (len(xgrid), len(ygrid)):
                    z_plot = z.T
                else:
                    print(f"Skip 2D PDP for ({f1}, {f2}) because unexpected z shape: {z.shape}")
                    continue

                pd_results[(i, j)] = (f1, f2, xgrid, ygrid, z_plot)
                zmins.append(np.nanmin(z_plot))
                zmaxs.append(np.nanmax(z_plot))

        if len(pd_results) == 0:
            print(f"No valid PDP interaction results for {suffix}.")
            return

        vmin = float(np.nanmin(zmins))
        vmax = float(np.nanmax(zmaxs))
        if np.isclose(vmin, vmax):
            vmin -= 1e-6
            vmax += 1e-6
        levels = np.linspace(vmin, vmax, 14)

        
        if suffix == "all_continuous_features":
            fig_size_cm = max(18, min(55, 2.15 * n + 5))
            tick_fontsize = 8 if n >= 10 else 9
            diag_fontsize = 9 if n >= 10 else 11
        else:
            fig_size_cm = max(12, min(24, 2.6 * n + 4))
            tick_fontsize = 9
            diag_fontsize = 11

        fig, axes = plt.subplots(
            n,
            n,
            figsize=(fig_size_cm * CM_TO_INCH, fig_size_cm * CM_TO_INCH),
            sharex="col",
            sharey="row"
        )

        if n == 1:
            axes = np.array([[axes]])
        elif isinstance(axes, plt.Axes):
            axes = np.array([[axes]])

        mappable = None
        abbr_names = [format_plot_label(abbr_feature_name(f) if "abbr_feature_name" in globals() else f) for f in features_subset]

        for i in range(n):
            for j in range(n):
                ax = axes[i, j]

                if i == j:
                    ax.text(
                        0.5, 0.5, abbr_names[i],
                        ha="center", va="center",
                        fontsize=diag_fontsize, fontname="Times New Roman",
                        wrap=True,
                        transform=ax.transAxes
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                elif i > j:
                    if (i, j) not in pd_results:
                        ax.axis("off")
                        continue

                    f1, f2, xgrid, ygrid, z_plot = pd_results[(i, j)]

                    contour = ax.contourf(
                        xgrid,
                        ygrid,
                        z_plot,
                        levels=levels,
                        cmap=COMMON_CMAP,
                        vmin=vmin,
                        vmax=vmax
                    )
                    mappable = contour

                    if j == 0:
                        ax.set_ylabel(abbr_names[i], fontsize=11, fontname="Times New Roman")
                        ax.tick_params(axis="y", labelsize=tick_fontsize)
                    else:
                        ax.set_ylabel("")
                        ax.tick_params(axis="y", labelleft=False)

                    if i == n - 1:
                        ax.set_xlabel(abbr_names[j], fontsize=11, fontname="Times New Roman")
                        ax.tick_params(axis="x", labelsize=tick_fontsize, rotation=30)
                    else:
                        ax.set_xlabel("")
                        ax.tick_params(axis="x", labelbottom=False)

                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_fontname("Times New Roman")

                else:
                    ax.axis("off")

        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
            cbar.set_label("Partial dependence", fontsize=11, fontname="Times New Roman")
            cbar.ax.tick_params(labelsize=11)
            for label in cbar.ax.get_yticklabels():
                label.set_fontname("Times New Roman")

        apply_times_new_roman(fig, font_size=11)
        fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.95, wspace=0.08, hspace=0.08)

        pdp_path = out_dir / f"{model_name}_2D_PDP_matrix_{suffix}.svg"
        save_figure_dual(fig, pdp_path)
        plt.close(fig)
        print("Saved 2D PDP matrix figure to:", pdp_path)

    
    _plot_one_pdp_matrix(
        candidate_features,
        "all_continuous_features",
        "all continuous PDP features"
    )

    
    for k in [4, 5, 6]:
        if len(candidate_features) >= k:
            _plot_one_pdp_matrix(
                candidate_features[:k],
                f"top_{k}",
                f"top {k} continuous PDP features"
            )
        else:
            print(f"Skip top_{k} PDP matrix: only {len(candidate_features)} continuous features are available.")


def compute_shap_for_best_model(best_pipe, model_name, X_background, X_explain):
    """Implementation details are documented in README.md and METHODS_AND_FORMULAE.md."""
    if shap is None:
        print("SHAP is not installed. Skip SHAP analysis.")
        return None, None, None

    preprocessor = best_pipe.named_steps["preprocess"]
    model = best_pipe.named_steps["model"]

    X_bg_trans = preprocessor.transform(X_background)
    X_exp_trans = preprocessor.transform(X_explain)

    feature_names = preprocessor.get_feature_names_out().tolist()
    X_exp_df = pd.DataFrame(X_exp_trans, columns=feature_names, index=X_explain.index)

    
    tree_models = ("AdaBoost", "RF", "GBDT", "XGBoost", "LGBM", "CatBoost")

    if model_name in tree_models:
        explainer = shap.Explainer(model, X_bg_trans, feature_names=feature_names)
        shap_values = explainer(X_exp_trans)
        values = shap_values.values
    else:
        
        bg = shap.sample(X_bg_trans, min(100, X_bg_trans.shape[0]), random_state=RANDOM_STATE)
        exp = X_exp_trans[:min(200, X_exp_trans.shape[0])]
        explainer = shap.KernelExplainer(model.predict, bg)
        values = explainer.shap_values(exp)
        X_exp_df = X_exp_df.iloc[:values.shape[0], :]

    return values, X_exp_df, feature_names


if shap is not None:
    
    X_bg = X_train.sample(min(200, len(X_train)), random_state=RANDOM_STATE)
    X_exp = X_test.copy()

    shap_values, X_exp_processed, feature_names = compute_shap_for_best_model(
        best_pipe,
        best_model_name,
        X_bg,
        X_exp
    )

    if shap_values is not None:
        
        shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_exp_processed.index)
        shap_df[PROVINCE_COL] = prov_test.loc[X_exp_processed.index].values
        shap_df["y_true"] = y_test.loc[X_exp_processed.index].values
        shap_df["y_pred"] = best_pred[:len(shap_df)]

        shap_values_path = SHAP_DIR / f"{best_model_name}_SHAP_values_by_sample.xlsx"
        shap_df.to_excel(shap_values_path, index=True)
        print("Saved SHAP values to:", shap_values_path)

        
        
        try:
            all_feature_columns_for_table = list(X.columns) + list(feature_names)
            save_feature_abbreviation_table(all_feature_columns_for_table, OUT_DIR)
        except Exception as e:
            print("Warning: failed to save feature abbreviation table:", repr(e))

        
        mean_abs_shap = (
            pd.DataFrame({
                "feature": feature_names,
                "mean_abs_SHAP": np.abs(shap_values).mean(axis=0)
            })
            .sort_values("mean_abs_SHAP", ascending=False)
            .reset_index(drop=True)
        )

        global_importance_path = SHAP_DIR / f"{best_model_name}_global_SHAP_importance.xlsx"
        mean_abs_shap.to_excel(global_importance_path, index=False)
        print("Saved global SHAP importance to:", global_importance_path)

        
        province_shap = shap_df.groupby(PROVINCE_COL)[feature_names].apply(
            lambda x: np.abs(x).mean(axis=0)
        ).reset_index()

        province_shap_path = SHAP_DIR / f"{best_model_name}_province_mean_abs_SHAP.xlsx"
        province_shap.to_excel(province_shap_path, index=False)
        print("Saved province-level SHAP to:", province_shap_path)

        
        save_shap_svg_plots(
            shap_values=shap_values,
            X_exp_processed=X_exp_processed,
            feature_names=feature_names,
            mean_abs_shap=mean_abs_shap,
            model_name=best_model_name,
            out_dir=FIG_DIR
        )

        
        save_region_shap_analysis(
            shap_values=shap_values,
            X_exp_processed=X_exp_processed,
            feature_names=feature_names,
            X_original_subset=X_test.loc[X_exp_processed.index].copy(),
            province_subset=prov_test,
            model_name=best_model_name,
            region_col=REGION_COL,
            base_out_dir=SHAP_DIR
        )

        
        save_pdp_matrix(
            best_pipe=best_pipe,
            X_train=X_train,
            mean_abs_shap=mean_abs_shap,
            out_dir=FIG_DIR,
            model_name=best_model_name,
            top_k=None,
            grid_resolution=20,
            max_features_per_matrix=8
        )

        
        long_prov = province_shap.melt(
            id_vars=[PROVINCE_COL],
            var_name="feature",
            value_name="mean_abs_SHAP"
        )
        top_features_by_province = (
            long_prov.sort_values([PROVINCE_COL, "mean_abs_SHAP"], ascending=[True, False])
            .groupby(PROVINCE_COL)
            .head(10)
            .reset_index(drop=True)
        )
        top_prov_path = SHAP_DIR / f"{best_model_name}_top10_SHAP_features_each_province.xlsx"
        top_features_by_province.to_excel(top_prov_path, index=False)
        print("Saved top province SHAP features to:", top_prov_path)

print("\nAll done.")
print("Output folder:", OUT_DIR.resolve())
