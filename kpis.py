# kpis.py
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


def _format_metric_value(title: str, value: object) -> str:
    # integer-like metrics
    if title in ("Count (valid)", "Missing", "Unique", "Unique categories", "Span (days)"):
        try:
            return str(int(value))  # type: ignore[arg-type]
        except Exception:
            return str(value)

    # numeric formatting
    if isinstance(value, (float, int)):
        return f"{value:,.2f}"

    return str(value)


def _render_metrics_grid(kpis: dict[str, object], col_name: str, *, max_items: int | None = None) -> None:
    """
    Render KPIs in rows of 3 using st.metric.
    """
    items = list(kpis.items())
    if max_items is not None:
        items = items[:max_items]

    for row_start in range(0, len(items), 3):
        row = items[row_start : row_start + 3]
        cols = st.columns(3)
        for i, (title, result) in enumerate(row):
            with cols[i]:
                st.metric(label=f"{title} of {col_name}", value=_format_metric_value(title, result))


def _detect_date_like(
    s0: pd.Series,
    *,
    semantic: Optional[str],
    sem_date_value: str,
    to_datetime_fn: Callable[[pd.Series], pd.Series],
) -> tuple[Optional[pd.Series], bool]:
    """
    Decide if a series is date-like.
    Returns (datetime_series_or_none, is_date_like).
    """
    # semantic override
    if semantic == sem_date_value:
        dt = to_datetime_fn(s0)
        return dt, dt.notna().any()

    # already datetime dtype
    if is_datetime64_any_dtype(s0):
        return s0, True

    # object -> try parsing and check ratio
    if is_object_dtype(s0):
        dt_candidate = to_datetime_fn(s0)
        is_date_like = dt_candidate.notna().mean() > 0.6
        return (dt_candidate if is_date_like else None), is_date_like

    return None, False


def numeric_kpis(s: pd.Series) -> dict[str, object]:
    """
    Pure KPI computation for numeric series.
    """
    valid_count = int(s.notna().sum())
    missing_count = int(s.isna().sum())

    return {
        "Count (valid)": valid_count,
        "Missing": missing_count,
        "Sum": s.sum(),
        "Min": s.min(),
        "Max": s.max(),
        "Mean": s.mean(),
    }


def date_kpis(dt: pd.Series, *, semantic: Optional[str], sem_date_value: str) -> dict[str, object] | None:
    """
    Pure KPI computation for date/datetime series.
    Returns None if no valid dates.
    """
    valid = dt.dropna()
    if valid.empty:
        return None

    min_dt = valid.min()
    max_dt = valid.max()
    span_days = int((max_dt - min_dt).days)
    unique_dates = int(valid.dt.date.nunique())
    missing_count = int(dt.isna().sum())

    min_disp = min_dt.date() if semantic == sem_date_value else min_dt
    max_disp = max_dt.date() if semantic == sem_date_value else max_dt

    return {
        "Count (valid)": int(valid.shape[0]),
        "Missing": missing_count,
        "Unique": unique_dates,
        "Min": min_disp,
        "Max": max_disp,
        "Span (days)": span_days,
    }


def categorical_kpis(s: pd.Series) -> dict[str, object]:
    """
    Pure KPI computation for categorical series.
    """
    return {
        "Count (valid)": int(s.notna().sum()),
        "Missing": int(s.isna().sum()),
        "Unique categories": int(s.nunique(dropna=True)),
    }


def render_kpi_section(
    df: pd.DataFrame,
    *,
    columns: list[str],
    column_semantics: dict[str, str],
    sem_date_value: str,
    to_datetime_fn: Callable[[pd.Series], pd.Series],
    selectbox_key: str = "kpi_field_auto",
) -> None:
    """
    Streamlit KPI UI.

    UX goal: show the info box ABOVE the selectbox only while no field is selected
    (same placeholder pattern you used elsewhere).
    """
    st.subheader("🔢 Key Performance Indicators (KPIs)")

    # Placeholder ABOVE the selectbox so we can show/hide the info message cleanly
    info_slot = st.empty()

    kpi_column = st.selectbox(
        "Select the field for KPI calculation",
        options=columns,
        index=None,
        key=selectbox_key,
    )

    # Show info only until the user selects a field
    if kpi_column is None:
        info_slot.info("Select a data field below to view its key statistics.")
        return
    else:
        info_slot.empty()

    st.caption(f"Summary statistics for: {kpi_column}")

    s0 = df[kpi_column]
    semantic = column_semantics.get(kpi_column)

    # -------------------------
    # numeric
    # -------------------------
    if is_numeric_dtype(s0):
        _render_metrics_grid(numeric_kpis(s0), kpi_column)
        return

    # -------------------------
    # date-like detection + KPIs
    # -------------------------
    dt, is_date_like = _detect_date_like(
        s0,
        semantic=semantic,
        sem_date_value=sem_date_value,
        to_datetime_fn=to_datetime_fn,
    )
    if is_date_like and dt is not None:
        k = date_kpis(dt, semantic=semantic, sem_date_value=sem_date_value)
        if k is None:
            st.warning(
                f"Column '{kpi_column}' looks like a date/datetime field, but no valid dates could be parsed."
            )
            return
        _render_metrics_grid(k, kpi_column)
        return

    # -------------------------
    # category-like (category + text)
    # -------------------------
    if is_categorical_dtype(s0) or is_object_dtype(s0) or is_string_dtype(s0):
        _render_metrics_grid(categorical_kpis(s0), kpi_column, max_items=3)
        return
