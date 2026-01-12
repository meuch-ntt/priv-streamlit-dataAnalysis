# main.py
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import data_loader
import plotting
import kpis

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

# ==============================================================================
# Constants (single source of truth for "magic strings")
# ==============================================================================

# File extensions
EXT_CSV = ".csv"
EXT_XLSX = ".xlsx"
DATA_FILE_EXTS = (EXT_CSV, EXT_XLSX)

# Semantic override values (stored per dataset)
SEM_DATE = "date"

# UI target types (what the user selects)
TARGET_NUMBER = "number"
TARGET_DATETIME = "datetime"
TARGET_DATE = SEM_DATE  # date-only semantics stored separately; underlying dtype stays datetime normalized
TARGET_CATEGORY = "category"
TARGET_TEXT = "text"
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
# Derived df (apply overrides WITHOUT mutating raw_df)
# ==============================================================================

def _freeze_dict(d: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Makes a dict hashable/stable for caching."""
    return tuple(sorted(d.items()))


def apply_type_overrides(
    raw_df: pd.DataFrame,
    type_overrides: dict[str, str],
    column_semantics: dict[str, str],
) -> pd.DataFrame:
    """
    Returns a derived df with user-requested type conversions applied.
    Does NOT mutate raw_df.
    Uses copy(deep=False) so only converted columns allocate new arrays.
    """
    if not type_overrides and not column_semantics:
        return raw_df

    df2 = raw_df.copy(deep=False)

    for col, t in type_overrides.items():
        if col not in df2.columns:
            continue

        s = df2[col]

        if t == TARGET_CATEGORY:
            df2[col] = s.astype("category")

        elif t == TARGET_DATETIME:
            dt = pd.to_datetime(s, errors="coerce")
            if column_semantics.get(col) == SEM_DATE:
                dt = dt.dt.normalize()
            df2[col] = dt

        elif t == TARGET_NUMBER:
            num = pd.to_numeric(s, errors="coerce")
            non_null = num.dropna()

            if non_null.empty:
                df2[col] = num
            else:
                all_integers = (non_null.round(0) == non_null).all()
                df2[col] = num.astype("Int64") if all_integers else num

        elif t == TARGET_TEXT:
            df2[col] = s.astype("string")

    return df2


@st.cache_data(show_spinner=False)
def to_datetime_cached(s: pd.Series) -> pd.Series:
    """
    Cached datetime parsing to avoid re-parsing the same column multiple times
    (preview, KPI date detection, date line charts).
    """
    return pd.to_datetime(s, errors="coerce")


@st.cache_data(show_spinner=True)
def load_derived_df_cached(
    path: str,
    ext: str,
    sheet_name: str | None,
    token: tuple[float, int],
    frozen_type_overrides: tuple[tuple[str, str], ...],
    frozen_column_semantics: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    """
    Cached entry point for: raw load + applying overrides.
    Cache invalidates when file changes (token) OR overrides/semantics change.
    """
    raw_df = data_loader.load_raw_df_cached(path, ext, sheet_name, token)
    type_overrides = dict(frozen_type_overrides)
    column_semantics = dict(frozen_column_semantics)
    return apply_type_overrides(raw_df, type_overrides, column_semantics)


# ==============================================================================
# App setup
# ==============================================================================

st.set_page_config(page_title="IDAA", layout="centered", page_icon="📊")
st.title("📈  Inelligent Data Analyser APP")

# ==============================================================================
# Selecting the file
# ==============================================================================

working_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = f"{working_dir}/data"

if not os.path.isdir(folder_path):
    st.error(f"{MSG_FOLDER_NOT_FOUND_PREFIX} {folder_path}")
    st.stop()

files = data_loader.list_data_files(folder_path, DATA_FILE_EXTS)

if not files:
    st.warning(MSG_NO_FILES_FOUND)
    st.stop()

selected_file = st.selectbox("Select a file", files, index=None)

if not selected_file:
    st.info("First, select the file with the data you want to analyze.")
    st.stop()

file_path = os.path.join(folder_path, selected_file)
token = data_loader.file_token(file_path)
ext = EXT_XLSX if selected_file.endswith(EXT_XLSX) else EXT_CSV

# ---- Lightweight state only ----
if "loaded_file" not in st.session_state:
    st.session_state.loaded_file = None
if "loaded_sheet" not in st.session_state:
    st.session_state.loaded_sheet = None

# ---- Persist overrides PER dataset ----
if "overrides_by_dataset" not in st.session_state:
    st.session_state.overrides_by_dataset = {}


def get_dataset_state(dataset_key: tuple[str, str | None]):
    """Per-dataset storage for type overrides and semantic overrides."""
    entry = st.session_state.overrides_by_dataset.get(dataset_key)
    if entry is None:
        entry = {"type_overrides": {}, "column_semantics": {}}
        st.session_state.overrides_by_dataset[dataset_key] = entry
    return entry["type_overrides"], entry["column_semantics"]


# ==============================================================================
# Dataset selection (sheet) + cached df load
# ==============================================================================

sheet_name: str | None = None

if ext == EXT_XLSX:
    try:
        sheet_names = data_loader.list_excel_sheets_cached(file_path, token)
        if not sheet_names:
            st.error("No sheets found in this Excel file.")
            st.stop()

        sheet_name = st.selectbox("Select a sheet", sheet_names, index=None)
        if sheet_name is None:
            st.stop()

        is_new_dataset = (
            st.session_state.loaded_file != selected_file
            or st.session_state.loaded_sheet != sheet_name
        )
        if is_new_dataset:
            st.session_state.loaded_file = selected_file
            st.session_state.loaded_sheet = sheet_name

    except Exception as e:
        st.error("❌ Failed to read the Excel file. Please check the file and selected sheet.")
        st.exception(e)
        st.stop()
else:
    is_new_dataset = (
        st.session_state.loaded_file != selected_file
        or st.session_state.loaded_sheet is not None
    )
    if is_new_dataset:
        st.session_state.loaded_file = selected_file
        st.session_state.loaded_sheet = None

dataset_key = (selected_file, sheet_name)
type_overrides, column_semantics = get_dataset_state(dataset_key)

try:
    df = load_derived_df_cached(
        file_path,
        ext,
        sheet_name,
        token,
        _freeze_dict(type_overrides),
        _freeze_dict(column_semantics),
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
            s = to_datetime_cached(head_df_local[c])
            head_df_local[c] = s.dt.strftime("%Y-%m-%d")

    display_types = [_preview_type_for_column(current_df, c) for c in head_df_local.columns]
    display_types = [f"[{t.upper()}]" for t in display_types]
    dtypes_row_local = pd.Series(display_types, index=head_df_local.columns, name="Type")

    return pd.concat([pd.DataFrame(dtypes_row_local).T, head_df_local])


st.dataframe(render_preview(df))

# ==============================================================================
# Changing the Type
# ==============================================================================
st.subheader("🔧 Changing the Type")

# show info until user clicked "Change Type" at least once
if "change_type_has_clicked" not in st.session_state:
    st.session_state.change_type_has_clicked = False

info_slot = st.empty()
if not st.session_state.change_type_has_clicked:
    info_slot.info("Change the data type of columns if they were not interpreted correctly.")

if st.session_state.get("_reset_change_type_widgets", False):
    st.session_state["change_type_cols"] = []
    st.session_state["change_type_target"] = None
    st.session_state["_reset_change_type_widgets"] = False

col1, col2, col3 = st.columns([3, 3, 1.5])

with col1:
    choose_cols = st.multiselect(
        "Change the type of specific columns",
        options=columns,
        key="change_type_cols",
    )

with col2:
    target_type = st.selectbox(
        "Set type to:",
        options=TARGET_OPTIONS,
        index=None,
        placeholder="Choose an option",
        key="change_type_target",
    )

with col3:
    st.markdown("<div style='margin-bottom: 6px; font-weight: bold;'>Confirm</div>", unsafe_allow_html=True)
    change_type_clicked = st.button("Change Type")

if choose_cols and target_type and change_type_clicked:
    # user performed the action at least once -> hide info from now on
    st.session_state.change_type_has_clicked = True
    info_slot.empty()

    try:
        for col in choose_cols:
            if target_type == TARGET_DATE:
                type_overrides[col] = TARGET_DATETIME
                column_semantics[col] = SEM_DATE
            else:
                type_overrides[col] = target_type
                column_semantics.pop(col, None)

        st.session_state["change_type_success_msg"] = (
            f'✅ Column(s) "{", ".join(choose_cols)}" type changed successfully. See data preview above.'
        )
        st.session_state["change_type_success"] = True
        st.session_state["_reset_change_type_widgets"] = True
        st.rerun()

    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")

if st.session_state.get("change_type_success"):
    st.success(st.session_state["change_type_success_msg"])
    st.session_state["change_type_success"] = False

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
        to_datetime_fn=to_datetime_cached,
    )

# ==============================================================================
# Visualizations (still using plotting.py)
# ==============================================================================
elif section == "Visualizations":
    st.subheader("📊 Visualizations")

    # show info until user generated at least one plot
    if "viz_has_generated" not in st.session_state:
        st.session_state.viz_has_generated = False

    info_slot = st.empty()
    if not st.session_state.viz_has_generated:
        info_slot.info("Select the fields and type of plot you want to generate.")

    x_axis = st.selectbox("Select the X-axis", options=columns)
    x_s = df[x_axis]
    x_sem = column_semantics.get(x_axis)

    agg_func = None

    x_is_date_like = (x_sem == SEM_DATE) or is_datetime64_any_dtype(x_s)
    x_is_cat = is_categorical_dtype(x_s)

    # show Y-axis + Aggregation on same line when relevant
    if x_is_cat or x_is_date_like:
        col_y, col_agg = st.columns([2, 1])
        with col_y:
            y_axis = st.selectbox("Select the Y-axis", options=columns)
        with col_agg:
            agg_func = st.selectbox("Aggregation", options=AGG_OPTIONS)
    else:
        y_axis = st.selectbox("Select the Y-axis", options=columns)

    y_s = df[y_axis]
    y_is_numeric = is_numeric_dtype(y_s)

    plot_list = plotting.compatible_plots(
        x_is_categorical=x_is_cat,
        x_is_date_like=x_is_date_like,
        y_is_numeric=y_is_numeric,
        agg_func=agg_func,
        agg_sum_value=AGG_SUM,
    )

    if not plot_list:
        st.warning("No compatible plots for the selected columns.")
        st.stop()

    plot_type = st.selectbox("Select the type of plot", options=plot_list)

    generate_clicked = st.button("Generate Plot")

    if generate_clicked:
        # user has taken the action at least once -> hide info from now on
        st.session_state.viz_has_generated = True
        info_slot.empty()

        agg_part = f" ({agg_func})" if agg_func else ""

        # ✅ caption like in KPI section
        st.caption(f"{plot_type} of{agg_part}  {y_axis} by {x_axis}")

        try:
            if plot_type == plotting.PLOT_BAR:
                if agg_func is None:
                    st.error("Please choose an aggregation function before generating the bar chart.")
                    st.stop()

                fig = plotting.make_bar_chart(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    agg_func=agg_func,
                    agg_sum_value=AGG_SUM,
                    agg_avg_value=AGG_AVERAGE,
                )

            elif plot_type == plotting.PLOT_PIE:
                fig = plotting.make_pie_chart(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                )

            elif plot_type == plotting.PLOT_LINE:
                if not y_is_numeric:
                    st.error("Line Chart requires a numeric Y-axis.")
                    st.stop()
                if agg_func is None:
                    st.error("Please choose an aggregation function (sum or average).")
                    st.stop()

                fig = plotting.make_line_chart(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    agg_func=agg_func,
                    x_semantic=x_sem,
                    sem_date_value=SEM_DATE,
                    agg_sum_value=AGG_SUM,
                    agg_avg_value=AGG_AVERAGE,
                    to_datetime_fn=to_datetime_cached,
                )
            else:
                st.error(f"Unsupported plot type: {plot_type}")
                st.stop()

            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"❌ Failed to generate plot: {e}")
            st.exception(e)


