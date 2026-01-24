# data_transforms.py
from __future__ import annotations

import pandas as pd
import streamlit as st

# ==============================================================================
# Internal tokens (single source of truth)
# ==============================================================================

SEM_DATE = "date"

TARGET_NUMBER = "number"
TARGET_DATETIME = "datetime"
TARGET_CATEGORY = "category"
TARGET_TEXT = "text"


# ==============================================================================
# Helpers
# ==============================================================================

def freeze_dict(d: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Turn a dict into a stable, hashable value for cache keys."""
    return tuple(sorted(d.items()))


# ==============================================================================
# Type overrides (derived df)
# ==============================================================================

def apply_type_overrides(
    raw_df: pd.DataFrame,
    type_overrides: dict[str, str],
    column_semantics: dict[str, str],
) -> pd.DataFrame:
    """
    Create a derived dataframe from raw_df according to user's conversions/overrides
    WITHOUT mutating raw_df.
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
    """Cached datetime parsing for in-memory Series."""
    return pd.to_datetime(s, errors="coerce")


@st.cache_data(show_spinner=True)
def load_derived_df_cached(
    path: str,
    ext: str,
    sheet_name: str | None,
    token: tuple[float, int],
    frozen_type_overrides: tuple[tuple[str, str], ...],
    frozen_column_semantics: tuple[tuple[str, str], ...],
    *,
    _load_raw_df_fn,  # Streamlit ignores underscore args for hashing
) -> pd.DataFrame:
    """
    Cached derived load entry point:
    - loads raw df (cached by loader)
    - applies type overrides deterministically
    """
    raw_df = _load_raw_df_fn(path, ext, sheet_name, token)
    type_overrides = dict(frozen_type_overrides)
    column_semantics = dict(frozen_column_semantics)
    return apply_type_overrides(raw_df, type_overrides, column_semantics)
