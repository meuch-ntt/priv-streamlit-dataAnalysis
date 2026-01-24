# plotting.py
from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pandas.api.types import is_categorical_dtype, is_datetime64_any_dtype, is_numeric_dtype


# Plot names (internal)
PLOT_BAR = "Bar Chart"
PLOT_PIE = "Pie Chart"
PLOT_LINE = "Line Chart"
PLOT_CUMULATIVE_LINE = "Cumulative Line"
PLOT_SCATTER = "Scatter Plot"
PLOT_HIST = "Distribution"

# Business-friendly labels (UI only)
PLOT_LABELS = {
    PLOT_BAR: "Compare values across categories",
    PLOT_PIE: "Share of total",
    PLOT_LINE: "Trend over time",
    PLOT_CUMULATIVE_LINE: "Growth over time (cumulative)",
    PLOT_SCATTER: "Relationship between two metrics",
    PLOT_HIST: "Value distribution",
}


def compatible_plots(
    *,
    x_is_categorical: bool,
    x_is_date_like: bool,
    x_is_numeric: bool,
    y_is_numeric: bool,
    agg_func: Optional[str],
    agg_sum_value: str,
) -> list[str]:
    """
    Decide which plots are compatible with the current x/y selection.
    """
    plots: list[str] = []

    # Category × Numeric
    if x_is_categorical and y_is_numeric:
        plots.append(PLOT_BAR)
        if agg_func == agg_sum_value:
            plots.append(PLOT_PIE)

    # Date × Numeric
    if x_is_date_like and y_is_numeric:
        plots.append(PLOT_LINE)
        if agg_func == agg_sum_value:
            plots.append(PLOT_CUMULATIVE_LINE)

    return plots


def make_bar_chart(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_axis: str,
    agg_func: str,
    agg_sum_value: str,
    agg_avg_value: str,
) -> plt.Figure:
    import seaborn as sns  # lazy import

    fig, ax = plt.subplots(figsize=(6, 4))

    if agg_func == agg_sum_value:
        bar_df = df.groupby(x_axis)[y_axis].sum().reset_index(name=agg_sum_value)
        value_col = agg_sum_value
    elif agg_func == agg_avg_value:
        bar_df = df.groupby(x_axis)[y_axis].mean().reset_index(name=agg_avg_value)
        value_col = agg_avg_value
    else:
        raise ValueError(f"Unsupported aggregation: {agg_func}")

    bar_df = bar_df.sort_values(by=value_col, ascending=False)

    # Ensure the x-axis renders in descending order of the aggregated value
    x_order = bar_df[x_axis].astype(str).tolist()
    bar_df[x_axis] = bar_df[x_axis].astype(str)

    sns.barplot(x=bar_df[x_axis], y=bar_df[value_col], ax=ax, order=x_order)
    ax.tick_params(axis="x", rotation=45)

    ymax = bar_df[value_col].max()
    if pd.notna(ymax) and ymax != 0:
        ax.set_ylim(top=ymax * 1.10)

    # Value labels
    for p in ax.patches:
        height = p.get_height()
        if pd.isna(height):
            continue

        if agg_func == agg_avg_value:
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

    ax.tick_params(axis="y", labelsize=10)
    ax.set_title(f"{PLOT_BAR} of {y_axis} by {x_axis}", fontsize=12)
    ax.set_xlabel(x_axis, fontsize=10)
    ax.set_ylabel(y_axis, fontsize=10)

    fig.tight_layout()
    return fig


def make_pie_chart(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_axis: str,
    agg_func: str,
    agg_sum_value: str,
) -> plt.Figure:
    import seaborn as sns  # lazy import

    if agg_func != agg_sum_value:
        raise ValueError("Pie chart supports sum only.")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    pie_data = df.groupby(x_axis)[y_axis].sum().sort_values(ascending=False)

    # Avoid clutter for tiny slices
    def autopct_fn(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= 3 else ""

    wedges, _, _ = ax.pie(
        pie_data.values,
        labels=None,
        autopct=autopct_fn,
        startangle=90,
        counterclock=False,
        pctdistance=0.7,
        textprops={"fontsize": 10},
        colors=sns.color_palette("colorblind", n_colors=len(pie_data)),
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )

    ax.axis("equal")
    ax.set_title(f"{PLOT_PIE} of {y_axis} by {x_axis}", fontsize=12)

    ax.legend(
        wedges,
        pie_data.index.astype(str),
        title=x_axis,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    fig.tight_layout()
    return fig


def make_line_chart(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_axis: str,
    x_is_date_like: bool,
    agg_func: Optional[str],
    x_semantic: Optional[str],
    sem_date_value: str,
    agg_sum_value: str,
    agg_avg_value: str,
    to_datetime_fn: Callable[[pd.Series], pd.Series],
) -> plt.Figure:
    import seaborn as sns  # lazy import

    fig, ax = plt.subplots(figsize=(6, 4))

    line_df = df[[x_axis, y_axis]].dropna().copy()

    if x_is_date_like:
        if not is_datetime64_any_dtype(line_df[x_axis]):
            line_df[x_axis] = to_datetime_fn(line_df[x_axis])

        if x_semantic == sem_date_value:
            line_df[x_axis] = line_df[x_axis].dt.normalize()

        if agg_func == agg_sum_value:
            line_df = line_df.groupby(x_axis, as_index=False)[y_axis].sum()
        elif agg_func == agg_avg_value:
            line_df = line_df.groupby(x_axis, as_index=False)[y_axis].mean()
        else:
            raise ValueError(f"Unsupported aggregation: {agg_func}")

        line_df = line_df.sort_values(by=x_axis)
        sns.lineplot(x=line_df[x_axis], y=line_df[y_axis], ax=ax)
        ax.tick_params(axis="x", rotation=45)

    else:
        if agg_func == agg_sum_value:
            line_df = line_df.groupby(x_axis, as_index=False)[y_axis].sum()
        elif agg_func == agg_avg_value:
            line_df = line_df.groupby(x_axis, as_index=False)[y_axis].mean()
        else:
            raise ValueError(f"Unsupported aggregation: {agg_func}")

        line_df = line_df.sort_values(by=x_axis)
        sns.lineplot(x=line_df[x_axis], y=line_df[y_axis], ax=ax)

    ax.tick_params(axis="y", labelsize=10)
    ax.set_title(f"{PLOT_LINE} of {y_axis} by {x_axis}", fontsize=12)
    ax.set_xlabel(x_axis, fontsize=10)
    ax.set_ylabel(y_axis, fontsize=10)

    fig.tight_layout()
    return fig


def make_cumulative_line_chart(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_axis: str,
    x_semantic: Optional[str],
    sem_date_value: str,
    to_datetime_fn: Callable[[pd.Series], pd.Series],
) -> plt.Figure:
    import seaborn as sns  # lazy import

    fig, ax = plt.subplots(figsize=(6, 4))

    line_df = df[[x_axis, y_axis]].dropna().copy()

    if not is_datetime64_any_dtype(line_df[x_axis]):
        line_df[x_axis] = to_datetime_fn(line_df[x_axis])

    if x_semantic == sem_date_value:
        line_df[x_axis] = line_df[x_axis].dt.normalize()

    line_df = line_df.groupby(x_axis, as_index=False)[y_axis].sum().sort_values(by=x_axis)
    line_df["cumulative"] = line_df[y_axis].cumsum()

    sns.lineplot(x=line_df[x_axis], y=line_df["cumulative"], ax=ax)
    ax.tick_params(axis="x", rotation=45)

    ax.set_title(f"{PLOT_CUMULATIVE_LINE} of {y_axis} by {x_axis}", fontsize=12)
    ax.set_xlabel(x_axis, fontsize=10)
    ax.set_ylabel(f"Cumulative {y_axis}", fontsize=10)

    fig.tight_layout()
    return fig


def make_scatter_plot(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_axis: str,
    hue: Optional[str] = None,
) -> plt.Figure:
    import seaborn as sns  # lazy import

    fig, ax = plt.subplots(figsize=(6, 4))

    cols = [x_axis, y_axis] + ([hue] if hue else [])
    scatter_df = df[cols].dropna().copy()

    sns.scatterplot(
        data=scatter_df,
        x=x_axis,
        y=y_axis,
        hue=hue if hue else None,
        ax=ax,
    )

    ax.set_title(f"{PLOT_SCATTER}: {y_axis} vs {x_axis}", fontsize=12)
    ax.set_xlabel(x_axis, fontsize=10)
    ax.set_ylabel(y_axis, fontsize=10)

    if hue:
        ax.legend(title=hue, loc="best", frameon=False)

    fig.tight_layout()
    return fig


def make_histogram(df: pd.DataFrame, *, y_axis: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))

    vals = df[y_axis].dropna()
    ax.hist(vals, bins=20)

    ax.set_title(f"{PLOT_HIST} of {y_axis}", fontsize=12)
    ax.set_xlabel(y_axis, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)

    fig.tight_layout()
    return fig


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_numeric_dtype(df[c])]


def _category_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_categorical_dtype(df[c])]


def _date_columns(df: pd.DataFrame, column_semantics: dict[str, str], sem_date_value: str) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        s = df[c]
        is_date = (column_semantics.get(c) == sem_date_value) or is_datetime64_any_dtype(s)
        if is_date:
            cols.append(c)
    return cols


def _auto_index_for_single_option(options: list[str]) -> Optional[int]:
    """
    Return an index for auto-selection when only one option exists.
    """
    return 0 if len(options) == 1 else None


def render_visualizations_section(
    df: pd.DataFrame,
    *,
    columns: list[str],
    column_semantics: dict[str, str],
    sem_date_value: str,
    agg_options: list[str],
    agg_sum_value: str,
    agg_avg_value: str,
    to_datetime_fn: Callable[[pd.Series], pd.Series],
    x_key: str = "viz_x_axis",
    y_key: str = "viz_y_axis",
    agg_key: str = "viz_agg",
    plot_key: str = "viz_plot_type",
) -> None:
    """
    Streamlit visualization UI.
    """
    st.subheader("📊 Visualizations")

    # show info until user generated at least one plot
    if "viz_has_generated" not in st.session_state:
        st.session_state.viz_has_generated = False

    info_slot = st.empty()
    if not st.session_state.viz_has_generated:
        info_slot.info("Choose what to measure and how to break it down, then generate a chart.")

    numeric_cols = _numeric_columns(df)
    cat_cols = _category_columns(df)
    date_cols = _date_columns(df, column_semantics, sem_date_value)

    if not numeric_cols:
        st.warning("No numeric fields available to measure.")
        st.stop()

    mode = st.radio(
        "What do you want to analyze?",
        options=["Compare categories", "Trend over time", "Relationship between two metrics"],
        horizontal=True,
    )

    if mode == "Compare categories":
        st.caption("Compare totals/averages across groups like Region, Product, Customer Segment.")
    elif mode == "Trend over time":
        st.caption("See how a metric changes over time (daily/weekly/monthly).")
    else:
        st.caption("See if two metrics move together (correlation/outliers).")

    # ------------------------------------------------------------------
    # Mode: Relationship (Scatter)
    # ------------------------------------------------------------------
    if mode == "Relationship between two metrics":
        colx, coly = st.columns(2)

        with colx:
            x_metric = st.selectbox(
                "Metric for x-axis",
                options=numeric_cols,
                index=_auto_index_for_single_option(numeric_cols),
                placeholder="Choose a numeric field",
                help="Numeric metric (usually independent / explanatory variable).",
                key="scatter_x_metric",
            )
        with coly:
            y_metric = st.selectbox(
                "Metric for y-axis",
                options=numeric_cols,
                index=_auto_index_for_single_option(numeric_cols),
                placeholder="Choose a numeric field",
                help="Numeric metric (usually dependent / outcome variable).",
                key="scatter_y_metric",
            )

        hue = st.selectbox(
            "Optional: breakdown by category",
            options=[None] + cat_cols,
            index=0,
            key="scatter_hue",
        )

        ready = (x_metric is not None) and (y_metric is not None)

        if ready:
            generate_clicked = st.button("Generate Chart")
        else:
            generate_clicked = False

        if generate_clicked:
            st.session_state.viz_has_generated = True
            info_slot.empty()

            st.caption(f"{PLOT_LABELS[PLOT_SCATTER]}: {y_metric} vs {x_metric}")

            try:
                fig = make_scatter_plot(df, x_axis=x_metric, y_axis=y_metric, hue=hue)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"❌ Failed to generate chart: {e}")
                st.exception(e)

        return

    # ------------------------------------------------------------------
    # Modes: Compare categories / Trend over time
    # (layout kept the same as your summarize flow)
    # ------------------------------------------------------------------
    col_y, col_agg = st.columns([0.6, 0.4])

    with col_y:
        y_axis = st.selectbox(
            "Metric",
            options=numeric_cols,
            index=_auto_index_for_single_option(numeric_cols),
            placeholder="Choose a numeric field",
            help="What you want to analyze (y-axis): numeric value to analyze (e.g., Sales, Profit, Quantity).",
            key=y_key,
        )

    with col_agg:
        agg_func = st.selectbox(
            "Measure as",
            options=agg_options,
            index=_auto_index_for_single_option(agg_options),
            placeholder="Choose",
            format_func=lambda v: "Total" if v == agg_sum_value else ("Average" if v == agg_avg_value else str(v)),
            help="Calculation you want to peform on the numerical field",
            key=agg_key,
        )

    if mode == "Compare categories":
        if not cat_cols:
            st.warning("No category fields available to compare.")
            st.stop()

        x_axis = st.selectbox(
            "Breakdown: How do you want to break the measure down?",
            options=cat_cols,
            index=_auto_index_for_single_option(cat_cols),
            placeholder="Choose a category",
            help="Pick a field to group results by (x-axis, e.g., Region, Product, Segment).",
            key=x_key,
        )

    else:  # Trend over time
        if not date_cols:
            st.warning("No date fields available for trends over time.")
            st.stop()

        x_axis = st.selectbox(
            "Date field",
            options=date_cols,
            index=_auto_index_for_single_option(date_cols),
            placeholder="Choose a date field",
            help="Pick a date field to trend over time (e.g., Order Date, Created At).",
            key=x_key,
        )

    ready = (y_axis is not None) and (agg_func is not None) and (x_axis is not None)

    plot_type: Optional[str] = None
    x_is_date_like = False
    x_sem: Optional[str] = None

    if ready:
        y_s = df[y_axis]
        y_is_numeric = is_numeric_dtype(y_s)

        x_s = df[x_axis]
        x_sem = column_semantics.get(x_axis)

        x_is_date_like = (x_sem == sem_date_value) or is_datetime64_any_dtype(x_s)
        x_is_cat = is_categorical_dtype(x_s)
        x_is_num = is_numeric_dtype(x_s)

        plot_list = compatible_plots(
            x_is_categorical=x_is_cat,
            x_is_date_like=x_is_date_like,
            x_is_numeric=x_is_num,
            y_is_numeric=y_is_numeric,
            agg_func=agg_func,
            agg_sum_value=agg_sum_value,
        )

        if not plot_list:
            st.warning("No compatible charts for the selected fields.")
            return

        if len(plot_list) == 1:
            plot_type = plot_list[0]
        else:
            plot_type = st.selectbox(
                "Chart type",
                options=plot_list,
                index=_auto_index_for_single_option(plot_list),
                format_func=lambda p: PLOT_LABELS.get(p, p),
                placeholder="Choose a chart",
                key=plot_key,
            )

    ready_to_generate = ready and (plot_type is not None)

    if ready_to_generate:
        generate_clicked = st.button("Generate Chart")
    else:
        generate_clicked = False

    if generate_clicked:
        st.session_state.viz_has_generated = True
        info_slot.empty()

        agg_part = f" ({agg_func})" if agg_func else ""
        st.caption(f"{PLOT_LABELS.get(plot_type, plot_type)}{agg_part}: {y_axis} by {x_axis}")

        try:
            if plot_type == PLOT_BAR:
                fig = make_bar_chart(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    agg_func=agg_func,
                    agg_sum_value=agg_sum_value,
                    agg_avg_value=agg_avg_value,
                )

            elif plot_type == PLOT_PIE:
                fig = make_pie_chart(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    agg_func=agg_func,
                    agg_sum_value=agg_sum_value,
                )

            elif plot_type == PLOT_LINE:
                fig = make_line_chart(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    x_is_date_like=x_is_date_like,
                    agg_func=agg_func,
                    x_semantic=x_sem,
                    sem_date_value=sem_date_value,
                    agg_sum_value=agg_sum_value,
                    agg_avg_value=agg_avg_value,
                    to_datetime_fn=to_datetime_fn,
                )

            elif plot_type == PLOT_CUMULATIVE_LINE:
                fig = make_cumulative_line_chart(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    x_semantic=x_sem,
                    sem_date_value=sem_date_value,
                    to_datetime_fn=to_datetime_fn,
                )

            else:
                st.error(f"Unsupported plot type: {plot_type}")
                st.stop()

            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"❌ Failed to generate chart: {e}")
            st.exception(e)
