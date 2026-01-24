# main.py
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

import dataset_selector
import data_loader
import data_transforms as transforms
import kpis
import plotting
import section_changing_types  # new: extracted "Changing the Type" UI

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

# ==============================================================================
# Set-up
# ==============================================================================

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# File extensions
EXT_CSV = ".csv"
EXT_XLSX = ".xlsx"
DATA_FILE_EXTS = (EXT_CSV, EXT_XLSX)

# Semantic override values (stored per dataset)
SEM_DATE = transforms.SEM_DATE

# UI target types (what the user selects)
TARGET_NUMBER = transforms.TARGET_NUMBER
TARGET_DATETIME = transforms.TARGET_DATETIME
TARGET_DATE = SEM_DATE  # date-only semantics stored separately; underlying dtype stays datetime normalized
TARGET_CATEGORY = transforms.TARGET_CATEGORY
TARGET_TEXT = transforms.TARGET_TEXT
TARGET_OPTIONS = [TARGET_NUMBER, TARGET_DATETIME, TARGET_DATE, TARGET_CATEGORY]

# Aggregations
AGG_SUM = "sum"
AGG_AVERAGE = "average"
AGG_OPTIONS = [AGG_SUM, AGG_AVERAGE]

# Messages
MSG_NO_DATA_LOADED = "No data loaded. Please select a valid file/sheet."
MSG_NO_FILES_FOUND = "No .csv or .xlsx files found in the data folder."
MSG_FOLDER_NOT_FOUND_PREFIX = "Data folder not found:"

# ==============================================================================
# Select file, read and cache data
# ==============================================================================

st.set_page_config(page_title="IDAA", layout="centered", page_icon="📊")
st.title("📈  Intelligent Data Analyser APP")

working_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = f"{working_dir}/data"

(
    selected_file,
    file_path,
    ext,
    sheet_name,
    token,
    dataset_key,
) = dataset_selector.select_dataset_ui(
    folder_path=folder_path,
    data_file_exts=DATA_FILE_EXTS,
    ext_csv=EXT_CSV,
    ext_xlsx=EXT_XLSX,
    msg_folder_not_found_prefix=MSG_FOLDER_NOT_FOUND_PREFIX,
    msg_no_files_found=MSG_NO_FILES_FOUND,
)

type_overrides, column_semantics = dataset_selector.get_dataset_state(dataset_key)

try:
    df = transforms.load_derived_df_cached(
        file_path,
        ext,
        sheet_name,
        token,
        transforms.freeze_dict(type_overrides),
        transforms.freeze_dict(column_semantics),
        _load_raw_df_fn=data_loader.load_raw_df_cached,
    )
except Exception as e:
    st.error(MSG_NO_DATA_LOADED)
    st.exception(e)
    st.stop()

columns = df.columns.tolist()
st.write("")

# ==============================================================================
# Data Preview
# ==============================================================================

st.header("📄 Data Preview")
st.write("This is a preview of the data with the first rows and current type interpretations of the columns.")


def _preview_type_for_column(current_df: pd.DataFrame, col: str) -> str:
    if column_semantics.get(col) == SEM_DATE:
        return TARGET_DATE

    s = current_df[col]
    if is_categorical_dtype(s):
        return TARGET_CATEGORY
    if is_datetime64_any_dtype(s):
        return TARGET_DATETIME
    if is_numeric_dtype(s):
        return TARGET_NUMBER
    return TARGET_TEXT


def render_preview(current_df: pd.DataFrame) -> pd.DataFrame:
    head_df_local = current_df.head().copy()

    for c in head_df_local.columns:
        if column_semantics.get(c) == SEM_DATE:
            s = transforms.to_datetime_cached(head_df_local[c])
            head_df_local[c] = s.dt.strftime("%Y-%m-%d")

    display_types = [_preview_type_for_column(current_df, c) for c in head_df_local.columns]
    display_types = [f"[{t.upper()}]" for t in display_types]
    dtypes_row_local = pd.Series(display_types, index=head_df_local.columns, name="Type")

    preview_df = pd.concat([pd.DataFrame(dtypes_row_local).T, head_df_local])

    # keep "Type" row, enumerate data rows starting at 1
    preview_df.index = ["Type"] + list(range(1, len(preview_df)))

    return preview_df


st.dataframe(render_preview(df))

# ==============================================================================
# Changing the Type
# ==============================================================================

section_changing_types.set(
    columns=columns,
    target_options=TARGET_OPTIONS,
    target_datetime_value=TARGET_DATETIME,
    target_date_value=TARGET_DATE,
    sem_date_value=SEM_DATE,
    type_overrides=type_overrides,
    column_semantics=column_semantics,
)

# ==============================================================================
# Data Analysis Selection
# ==============================================================================

st.header("🔍 Data Analysis")

info_slot = st.empty()

section = st.selectbox(
    "Analysis type",
    options=["Key Performance Indicators (KPIs)", "Visualizations"],
    index=None,
    placeholder="Select what you want to do next",
)

if section is None:
    info_slot.info("Choose the type of analysis you want to perform")
    st.stop()
else:
    info_slot.empty()

# ==============================================================================
# KPI SECTION (moved to kpis.py)
# ==============================================================================

if section == "Key Performance Indicators (KPIs)":
    kpis.render_kpi_section(
        df,
        columns=columns,
        column_semantics=column_semantics,
        sem_date_value=SEM_DATE,
        to_datetime_fn=transforms.to_datetime_cached,
    )

# ==============================================================================
# Visualizations (still using plotting.py)
# ==============================================================================

elif section == "Visualizations":
    plotting.render_visualizations_section(
        df,
        columns=columns,
        column_semantics=column_semantics,
        sem_date_value=SEM_DATE,
        agg_options=AGG_OPTIONS,
        agg_sum_value=AGG_SUM,
        agg_avg_value=AGG_AVERAGE,
        to_datetime_fn=transforms.to_datetime_cached,
    )
