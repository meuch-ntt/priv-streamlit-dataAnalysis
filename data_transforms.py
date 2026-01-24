# data_transforms.py
from __future__ import annotations

import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype,
    is_categorical_dtype,
)

# ==============================================================================
# Internal tokens (single source of truth)
# ==============================================================================

SEM_DATE = "date"

TARGET_NUMBER = "number"
TARGET_DATETIME = "datetime"
TARGET_CATEGORY = "category"
TARGET_TEXT = "text"

AUTO_CATEGORY_MAX_UNIQUE = 10
AUTO_DATE_PARSE_THRESHOLD = 0.8  # fraction of non-null values that must parse as datetime


# ==============================================================================
# Helpers
# ==============================================================================

def freeze_dict(d: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Turn a dict into a stable, hashable value for cache keys."""
    return tuple(sorted(d.items()))


@st.cache_data(show_spinner=False)
def to_datetime_cached(s: pd.Series) -> pd.Series:
    """Cached datetime parsing for in-memory Series."""
    return pd.to_datetime(s, errors="coerce")


def _has_any_time_component(dt: pd.Series) -> bool:
    """
    True if any timestamp has a non-00:00:00 time.
    Expects dt to be datetime64[ns] with NaT allowed.
    """
    dt = dt.dropna()
    if dt.empty:
        return False

    return (
        (dt.dt.hour != 0)
        | (dt.dt.minute != 0)
        | (dt.dt.second != 0)
        | (dt.dt.microsecond != 0)
    ).any()


def infer_types_and_semantics(df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """
    Infer types and (optional) date-only semantics for columns.

    Rules:
      - object/string with <= 10 unique non-null values -> category
      - object/string that is date-like -> datetime, plus SEM_DATE if no time component
      - numeric -> number
      - datetime -> datetime, plus SEM_DATE if no time component
    """
    inferred_types: dict[str, str] = {}
    inferred_semantics: dict[str, str] = {}

    for col in df.columns:
        s = df[col]

        # Strong types first
        if is_bool_dtype(s):
            inferred_types[col] = TARGET_CATEGORY
            continue

        if is_numeric_dtype(s):
            inferred_types[col] = TARGET_NUMBER
            continue

        if is_datetime64_any_dtype(s):
            inferred_types[col] = TARGET_DATETIME
            if not _has_any_time_component(s):
                inferred_semantics[col] = SEM_DATE
            continue

        if is_categorical_dtype(s):
            inferred_types[col] = TARGET_CATEGORY
            continue

        # Object / string-like: try datetime parse
        dt = to_datetime_cached(s)
        parsed_ratio = dt.notna().mean()

        if parsed_ratio >= AUTO_DATE_PARSE_THRESHOLD:
            inferred_types[col] = TARGET_DATETIME
            if not _has_any_time_component(dt):
                inferred_semantics[col] = SEM_DATE
            continue

        # Otherwise: low cardinality -> category, else text
        nunique = s.dropna().nunique()
        if nunique <= AUTO_CATEGORY_MAX_UNIQUE:
            inferred_types[col] = TARGET_CATEGORY
        else:
            inferred_types[col] = TARGET_TEXT

    return inferred_types, inferred_semantics


# ==============================================================================
# Derived DF (apply inference deterministically)
# ==============================================================================

def apply_inferred_types(
    raw_df: pd.DataFrame,
    inferred_types: dict[str, str],
    inferred_semantics: dict[str, str],
) -> pd.DataFrame:
    """
    Create a derived dataframe from raw_df according to inferred conversions
    WITHOUT mutating raw_df.
    """
    if not inferred_types and not inferred_semantics:
        return raw_df

    df2 = raw_df.copy(deep=False)

    for col, t in inferred_types.items():
        if col not in df2.columns:
            continue

        s = df2[col]

        if t == TARGET_CATEGORY:
            df2[col] = s.astype("category")

        elif t == TARGET_DATETIME:
            dt = to_datetime_cached(s)
            if inferred_semantics.get(col) == SEM_DATE:
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


@st.cache_data(show_spinner=True)
def load_derived_df_cached(
    path: str,
    ext: str,
    sheet_name: str | None,
    token: tuple[float, int],
    frozen_inferred_types: tuple[tuple[str, str], ...],
    frozen_inferred_semantics: tuple[tuple[str, str], ...],
    *,
    _load_raw_df_fn,  # Streamlit ignores underscore args for hashing
) -> pd.DataFrame:
    """
    Cached derived load entry point:
    - loads raw df (cached by loader)
    - infers types deterministically (passed in)
    - applies inference deterministically
    """
    raw_df = _load_raw_df_fn(path, ext, sheet_name, token)

    inferred_types = dict(frozen_inferred_types)
    inferred_semantics = dict(frozen_inferred_semantics)

    return apply_inferred_types(raw_df, inferred_types, inferred_semantics)
