import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import plotting

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
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

# Plot names
PLOT_BAR = "Bar Chart"
PLOT_PIE = "Pie Chart"
PLOT_LINE = "Line Chart"
PLOT_SCATTER = "Scatter Plot"

# Messages
MSG_NO_DATA_LOADED = "No data loaded. Please select a valid file/sheet."
MSG_NO_FILES_FOUND = "No .csv or .xlsx files found in the data folder."
MSG_FOLDER_NOT_FOUND_PREFIX = "Data folder not found:"


# ==============================================================================
# Caching helpers (file listing + file reads)
# ==============================================================================

@st.cache_data(show_spinner=False)
def list_data_files(folder_path: str, exts: tuple[str, ...]) -> list[str]:
    return sorted([f for f in os.listdir(folder_path) if f.endswith(exts)])


@st.cache_data(show_spinner=False)
def file_token(path: str) -> tuple[float, int]:
    # Cache-buster token: changes when file content changes.
    return (os.path.getmtime(path), os.path.getsize(path))


@st.cache_data(show_spinner=True)
def read_csv_cached(path: str, token: tuple[float, int]) -> pd.DataFrame:
    # token exists only to invalidate cache when file changes
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def list_excel_sheets_cached(path: str, token: tuple[float, int]) -> list[str]:
    with pd.ExcelFile(path) as xls:
        return xls.sheet_names


@st.cache_data(show_spinner=True)
def read_excel_sheet_cached(path: str, sheet_name: str, token: tuple[float, int]) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name)


# ==============================================================================
# Derived df (apply overrides WITHOUT mutating raw_df)
# ==============================================================================

def _freeze_dict(d: dict[str, str]) -> tuple[tuple[str, str], ...]:
    # Makes a dict hashable/stable for caching.
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
                # Robust integer detection (avoid float precision issues from modulo)
                all_integers = (non_null.round(0) == non_null).all()
                df2[col] = num.astype("Int64") if all_integers else num

        elif t == TARGET_TEXT:
            df2[col] = s.astype("string")

    return df2


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
    Single cached entry point for: raw load + applying overrides.
    Cache invalidates when file changes (token) OR overrides/semantics change.
    """
    if ext == EXT_XLSX:
        if sheet_name is None:
            raise ValueError("sheet_name is required for Excel files.")
        raw_df = read_excel_sheet_cached(path, sheet_name, token)
    else:
        raw_df = read_csv_cached(path, token)

    type_overrides = dict(frozen_type_overrides)
    column_semantics = dict(frozen_column_semantics)

    return apply_type_overrides(raw_df, type_overrides, column_semantics)


# ==============================================================================
# App setup
# ==============================================================================

st.set_page_config(page_title="Data Visualizer", layout="centered", page_icon="📊")
st.title("📈  Data Visualizer")

# ==============================================================================
# Selecting the file
# ==============================================================================

working_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = f"{working_dir}/data"

if not os.path.isdir(folder_path):
    st.error(f"{MSG_FOLDER_NOT_FOUND_PREFIX} {folder_path}")
    st.stop()

files = list_data_files(folder_path, DATA_FILE_EXTS)

if not files:
    st.warning(MSG_NO_FILES_FOUND)
    st.stop()

selected_file = st.selectbox("Select a file", files, index=None)

if not selected_file:
    st.info("Please select a file to continue.")
    st.stop()

file_path = os.path.join(folder_path, selected_file)
token = file_token(file_path)
ext = EXT_XLSX if selected_file.endswith(EXT_XLSX) else EXT_CSV

# ---- Lightweight state only ----
if "loaded_file" not in st.session_state:
    st.session_state.loaded_file = None
if "loaded_sheet" not in st.session_state:
    st.session_state.loaded_sheet = None

# ---- Persist overrides PER dataset ----
if "overrides_by_dataset" not in st.session_state:
    # dataset_key -> {"type_overrides": {...}, "column_semantics": {...}}
    st.session_state.overrides_by_dataset = {}


def get_dataset_state(dataset_key: tuple[str, str | None]):
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
        sheet_names = list_excel_sheets_cached(file_path, token)
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

# Cached derived df (raw load + overrides)
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

st.header("🔍 Data Preview")


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
            s = pd.to_datetime(head_df_local[c], errors="coerce")
            head_df_local[c] = s.dt.strftime("%Y-%m-%d")

    display_types = [_preview_type_for_column(current_df, c) for c in head_df_local.columns]
    display_types = [f"[{t.upper()}]" for t in display_types]
    dtypes_row_local = pd.Series(display_types, index=head_df_local.columns, name="Type")

    return pd.concat([pd.DataFrame(dtypes_row_local).T, head_df_local])


st.dataframe(render_preview(df))

# ==============================================================================
# Changing the Type (update overrides; cached derived df updates on rerun)
# ==============================================================================
import time 

st.header("🔧 Changing the Type")
st.info("Change the data type of columns if they were not interpreted correctly.")

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
        st.session_state["_reset_change_type_widgets"] = True
        st.rerun()

    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")

if "change_type_success_until" in st.session_state:
    st.success(
            st.session_state['change_type_success_msg']
        )
    
# ==============================================================================
# KPI SECTION
# ==============================================================================

st.header("🔢 Key Performance Indicators (KPIs)")
st.info("Select a data field below to view its key statistics.")

kpi_column = st.selectbox(
    "Select the field for KPI calculation",
    options=columns,
    index=None,
    key="kpi_field_auto",
)

if kpi_column is not None:
    s0 = df[kpi_column]
    semantic = column_semantics.get(kpi_column)

    # NUMERIC
    if is_numeric_dtype(s0):
        s = s0
        valid_count = int(s.notna().sum())
        missing_count = int(s.isna().sum())

        kpi_calculations = {
            "Count (valid)": valid_count,
            "Missing": missing_count,
            "Sum": s.sum(),
            "Min": s.min(),
            "Max": s.max(),
            "Mean": s.mean(),
        }

        kpi_list = list(kpi_calculations.items())

        c1, c2, c3 = st.columns(3)
        for i in range(3):
            title, result = kpi_list[i]
            if title in ("Count (valid)", "Missing"):
                display_value = str(int(result))
            elif isinstance(result, (float, int)):
                display_value = f"{result:,.2f}"
            else:
                display_value = str(result)
            with [c1, c2, c3][i]:
                st.metric(label=f"{title} of {kpi_column}", value=display_value)

        c4, c5, c6 = st.columns(3)
        for i in range(3):
            title, result = kpi_list[i + 3]
            if title in ("Count (valid)", "Missing"):
                display_value = str(int(result))
            elif isinstance(result, (float, int)):
                display_value = f"{result:,.2f}"
            else:
                display_value = str(result)
            with [c4, c5, c6][i]:
                st.metric(label=f"{title} of {kpi_column}", value=display_value)

    # DATE / DATETIME
    else:
        dt = None
        is_date_like = False

        if semantic == SEM_DATE:
            is_date_like = True
            dt = pd.to_datetime(s0, errors="coerce")

        elif is_datetime64_any_dtype(s0):
            is_date_like = True
            dt = s0

        elif is_object_dtype(s0):
            dt_candidate = pd.to_datetime(s0, errors="coerce")
            is_date_like = dt_candidate.notna().mean() > 0.6
            dt = dt_candidate if is_date_like else None

        if is_date_like and dt is not None:
            valid = dt.dropna()
            missing_count = int(dt.isna().sum())

            if valid.empty:
                st.warning(f"Column '{kpi_column}' looks like a date/datetime field, but no valid dates could be parsed.")
            else:
                min_dt = valid.min()
                max_dt = valid.max()
                span_days = int((max_dt - min_dt).days)
                unique_dates = int(valid.dt.date.nunique())

                min_disp = min_dt.date() if semantic == SEM_DATE else min_dt
                max_disp = max_dt.date() if semantic == SEM_DATE else max_dt

                kpi_calculations = {
                    "Count (valid)": int(valid.shape[0]),
                    "Missing": missing_count,
                    "Unique": unique_dates,
                    "Min": min_disp,
                    "Max": max_disp,
                    "Span (days)": span_days,
                }

                kpi_list = list(kpi_calculations.items())

                c1, c2, c3 = st.columns(3)
                for i in range(3):
                    title, result = kpi_list[i]
                    with [c1, c2, c3][i]:
                        st.metric(label=f"{title} of {kpi_column}", value=str(result))

                c4, c5, c6 = st.columns(3)
                for i in range(3):
                    title, result = kpi_list[i + 3]
                    with [c4, c5, c6][i]:
                        st.metric(label=f"{title} of {kpi_column}", value=str(result))

        elif is_categorical_dtype(s0):
            s = s0
            kpi_calculations = {
                "Count (valid)": int(s.notna().sum()),
                "Missing": int(s.isna().sum()),
                "Unique categories": int(s.nunique(dropna=True)),
            }
            kpi_list = list(kpi_calculations.items())
            c1, c2, c3 = st.columns(3)
            for i in range(3):
                title, result = kpi_list[i]
                with [c1, c2, c3][i]:
                    st.metric(label=f"{title} of {kpi_column}", value=str(result))
        else:
            st.warning(f"The column '{kpi_column}' is not a supported type for KPIs (dtype={df[kpi_column].dtype}).")

# ==============================================================================
# Charts & Visuals
# ==============================================================================

st.header("📊 Charts & Visuals")

x_axis = st.selectbox("Select the X-axis", options=columns)
y_axis = st.selectbox("Select the Y-axis", options=columns)

agg_func = None
value_col_name = None
plot_list: list[str] = []

x_s = df[x_axis]
y_s = df[y_axis]
x_sem = column_semantics.get(x_axis)

if is_categorical_dtype(x_s) and is_numeric_dtype(y_s):
    agg_func = st.selectbox("Select the aggregation function", options=AGG_OPTIONS)
    plot_list.append(PLOT_BAR)
    if agg_func == AGG_SUM:
        plot_list.append(PLOT_PIE)

elif is_numeric_dtype(x_s) and is_numeric_dtype(y_s):
    plot_list.extend([PLOT_SCATTER, PLOT_LINE])

elif (x_sem == SEM_DATE or is_datetime64_any_dtype(x_s)) and is_numeric_dtype(y_s):
    plot_list.append(PLOT_LINE)

if not plot_list:
    st.warning("No compatible plots for the selected columns.")
    st.stop()

plot_type = st.selectbox("Select the type of plot", options=plot_list)

if st.button("Generate Plot"):
    fig, ax = plt.subplots(figsize=(6, 4))

    if plot_type == PLOT_BAR:
        if agg_func is None:
            st.error("Please choose an aggregation function before generating the bar chart.")
            st.stop()

        if agg_func == AGG_SUM:
            bar_df = df.groupby(x_axis)[y_axis].sum().reset_index(name=AGG_SUM)
            value_col_name = AGG_SUM
        elif agg_func == AGG_AVERAGE:
            bar_df = df.groupby(x_axis)[y_axis].mean().reset_index(name=AGG_AVERAGE)
            value_col_name = AGG_AVERAGE
        else:
            st.error(f"Unsupported aggregation: {agg_func}")
            st.stop()

        bar_df = bar_df.sort_values(by=value_col_name, ascending=False)

        sns.barplot(x=bar_df[x_axis], y=bar_df[value_col_name], ax=ax)
        ax.tick_params(axis="x", rotation=90)

        ymax = bar_df[value_col_name].max()
        if pd.notna(ymax) and ymax != 0:
            ax.set_ylim(top=ymax * 1.10)

        for p in ax.patches:
            height = p.get_height()
            if pd.isna(height):
                continue

            if agg_func == AGG_AVERAGE:
                label = f"{height:.2f}"
            else:
                label = f"{int(height)}" if float(height).is_integer() else f"{height:.2f}"

            ax.annotate(
                label,
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
                xytext=(0, 3),
                textcoords="offset points",
            )

    elif plot_type == PLOT_PIE:
        pie_data = df.groupby(x_axis)[y_axis].sum()
        ax.pie(
            pie_data,
            labels=pie_data.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("pastel"),
        )

    elif plot_type == PLOT_LINE:
        if not is_numeric_dtype(df[y_axis]):
            st.error("Line Chart requires a numeric Y-axis.")
            st.stop()

        if column_semantics.get(x_axis) == SEM_DATE or is_datetime64_any_dtype(df[x_axis]):
            plotting.create_lineChart_Date(df, x_axis, y_axis, ax)
        else:
            sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)

    elif plot_type == PLOT_SCATTER:
        sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    plt.title(f"{plot_type} of {y_axis} vs {x_axis}", fontsize=12)
    plt.xlabel(x_axis, fontsize=10)
    plt.ylabel(y_axis, fontsize=10)

    st.pyplot(fig)
    plt.close(fig)
