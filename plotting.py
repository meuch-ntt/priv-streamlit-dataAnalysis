# plotting.py
from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pandas.api.types import is_categorical_dtype, is_datetime64_any_dtype, is_numeric_dtype


# ==============================================================================
# Plot names (internal)
# ==============================================================================
PLOT_BAR = "Bar Chart"
PLOT_PIE = "Pie Chart"
PLOT_LINE = "Line Chart"
PLOT_CUMULATIVE_LINE = "Cumulative Line"
PLOT_SCATTER = "Scatter Plot"
PLOT_HIST = "Distribution"
PLOT_TOPN_TABLE = "Summary Table"
PLOT_HEATMAP = "Heatmap"
PLOT_TREEMAP = "Treemap"  # NEW


# ==============================================================================
# Business-friendly labels (UI only)
# ==============================================================================
PLOT_LABELS = {
    PLOT_BAR: "Top categories (bar chart)",
    PLOT_TOPN_TABLE: "Top categories (table)",
    PLOT_PIE: "Share of total (pie chart)",
    PLOT_LINE: "Trend over time (line chart)",
    PLOT_CUMULATIVE_LINE: "Growth over time (cumulative)",
    PLOT_SCATTER: "Relationship between two metrics (correlation)",
    PLOT_HIST: "Value distribution",
    PLOT_HEATMAP: "Heatmap (limited to Top 20 categories)",
    # Treemap label is dynamic (depends on computed coverage_pct), so handled in UI
}


# ==============================================================================
# Treemap controls (from your testing script)
# ==============================================================================
TREEMAP_PARETO_SHARE = 0.80
TREEMAP_MAX_CATEGORIES = 85

TREEMAP_TOP_N1 = 10
TREEMAP_TOP_N2 = 40
TREEMAP_CHARS_TOP1 = 25
TREEMAP_CHARS_TOP2 = 15
TREEMAP_CHARS_REST = 10

TREEMAP_ROTATE_IF_WIDTH_BELOW = 5.0  # tweak: 4.0 fewer rotations, 6.0 more


def compatible_plots(
    *,
    x_is_categorical: bool,
    x_is_date_like: bool,
    x_is_numeric: bool,
    x_is_text: bool,
    y_is_numeric: bool,
    agg_func: Optional[str],
    agg_sum_value: str,
    can_heatmap: bool,
    can_treemap: bool,  # NEW
) -> list[str]:
    """
    Decide which plots are compatible with the current x/y selection.
    """
    plots: list[str] = []

    # Category-like (categorical) × Numeric
    if x_is_categorical and y_is_numeric:
        plots.append(PLOT_BAR)
        plots.append(PLOT_TOPN_TABLE)
        if agg_func == agg_sum_value:
            plots.append(PLOT_PIE)

    # Text × Numeric (NO bar chart)
    if x_is_text and y_is_numeric:
        plots.append(PLOT_TOPN_TABLE)
        if can_heatmap:
            plots.append(PLOT_HEATMAP)
        if can_treemap:
            plots.append(PLOT_TREEMAP)

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

    # Keep labels readable
    ax.tick_params(axis="x", rotation=90)

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


def make_category_summary_table(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_axis: str,
    agg_func: str,
    agg_sum_value: str,
    agg_avg_value: str,
    top_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create a category summary table:
    aggregated metric (y) using agg_func by category (x), sorted highest first.

    If top_rows is None, returns ALL rows (no limit).
    """
    if agg_func == agg_sum_value:
        out = df.groupby(x_axis)[y_axis].sum().reset_index(name=agg_sum_value)
        value_col = agg_sum_value
    elif agg_func == agg_avg_value:
        out = df.groupby(x_axis)[y_axis].mean().reset_index(name=agg_avg_value)
        value_col = agg_avg_value
    else:
        raise ValueError(f"Unsupported aggregation: {agg_func}")

    out[x_axis] = out[x_axis].astype(str)
    out = out.sort_values(by=value_col, ascending=False)

    if top_rows is not None:
        out = out.head(int(top_rows))

    out.insert(0, "Rank", range(1, len(out) + 1))
    return out


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
        ax.tick_params(axis="x", rotation=90)
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


def make_heatmap(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_axis: str,
    heatmap_by: str,
    agg_func: str,
    agg_sum_value: str,
    agg_avg_value: str,
    top_rows: int = 20,
    top_cols: int = 20,
) -> plt.Figure:
    import seaborn as sns  # lazy import

    work = df[[x_axis, heatmap_by, y_axis]].dropna().copy()
    work[x_axis] = work[x_axis].astype(str)
    work[heatmap_by] = work[heatmap_by].astype(str)

    if agg_func == agg_sum_value:
        grouped = work.groupby([x_axis, heatmap_by], as_index=False)[y_axis].sum()
        value_name = agg_sum_value
    elif agg_func == agg_avg_value:
        grouped = work.groupby([x_axis, heatmap_by], as_index=False)[y_axis].mean()
        value_name = agg_avg_value
    else:
        raise ValueError(f"Unsupported aggregation: {agg_func}")

    grouped = grouped.rename(columns={y_axis: value_name})

    pivot = grouped.pivot_table(
        index=x_axis,
        columns=heatmap_by,
        values=value_name,
        aggfunc="sum",
        fill_value=0,
    )

    row_order = pivot.sum(axis=1).sort_values(ascending=False).head(int(top_rows)).index
    col_order = pivot.sum(axis=0).sort_values(ascending=False).head(int(top_cols)).index
    pivot = pivot.loc[row_order, col_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, ax=ax, cmap="Blues")
    ax.set_title(f"{PLOT_HEATMAP}: {y_axis} by {x_axis} and {heatmap_by}", fontsize=12)
    ax.set_xlabel(heatmap_by, fontsize=10)
    ax.set_ylabel(x_axis, fontsize=10)
    fig.tight_layout()
    return fig


def make_treemap_pareto_text_breakdown(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_axis: str,
    agg_func: str,
    agg_sum_value: str,
    agg_avg_value: str,
    pareto_share: float = TREEMAP_PARETO_SHARE,
    max_categories: int = TREEMAP_MAX_CATEGORIES,
) -> tuple[plt.Figure, int]:
    """
    Treemap for text breakdowns:
    Shows up to `max_categories` groups, chosen as the smallest set that reaches `pareto_share`
    (capped), with the remainder aggregated as 'Rest'.

    Returns: (figure, coverage_pct)
    """
    import squarify  # lazy import

    if agg_func == agg_sum_value:
        agg = df.groupby(x_axis, as_index=False)[y_axis].sum()
    elif agg_func == agg_avg_value:
        agg = df.groupby(x_axis, as_index=False)[y_axis].mean()
    else:
        raise ValueError(f"Unsupported aggregation: {agg_func}")

    agg = (
        agg.rename(columns={y_axis: "value"})
           .dropna(subset=[x_axis, "value"])
    )
    agg[x_axis] = agg[x_axis].astype(str)
    agg = agg.sort_values("value", ascending=False).reset_index(drop=True)
    distinct_products = agg[x_axis].nunique()

    total_val = agg["value"].sum()
    if total_val == 0 or pd.isna(total_val):
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.text(0.5, 0.5, "No data to display.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig, 0

    agg["cum_share"] = agg["value"].cumsum() / total_val
    k = int((agg["cum_share"] < pareto_share).sum() + 1)
    top_n = min(k, max_categories)

    top_df = agg.head(top_n)[[x_axis, "value"]]
    rest_val = agg.iloc[top_n:]["value"].sum()

    coverage_pct = round((top_df["value"].sum() / total_val) * 100)

    treemap_df = pd.concat(
        [top_df, pd.DataFrame([{x_axis: "Rest", "value": rest_val}])],
        ignore_index=True,
    )

    # label truncation (same logic as your notebook)
    def _trunc(s: str, n: int) -> str:
        return (s[:n] + "…") if len(s) > n else s

    labels: list[str] = []
    for i, name in enumerate(treemap_df[x_axis].tolist()):
        if name == "Rest":
            labels.append("Rest")
        elif i < TREEMAP_TOP_N1:
            labels.append(_trunc(name, TREEMAP_CHARS_TOP1))
        elif i < TREEMAP_TOP_N2:
            labels.append(_trunc(name, TREEMAP_CHARS_TOP2))
        else:
            labels.append(_trunc(name, TREEMAP_CHARS_REST))

    # Draw using squarify.plot to preserve layout/colors, then overlay rotated labels
    fig = plt.figure(figsize=(16, 9))
    ax = plt.gca()

    squarify.plot(
        sizes=treemap_df["value"],
        label=None,  # overlay labels ourselves
        alpha=0.85,
        ax=ax,
    )

    sizes = treemap_df["value"].tolist()
    norm_sizes = squarify.normalize_sizes(sizes, 100, 100)
    rects = squarify.squarify(norm_sizes, 0, 0, 100, 100)

    for i, r in enumerate(rects):
        x, y, dx, dy = r["x"], r["y"], r["dx"], r["dy"]
        rotate = (labels[i] != "Rest" and dx < TREEMAP_ROTATE_IF_WIDTH_BELOW)

        ax.text(
            x + dx / 2,
            y + dy / 2,
            labels[i],
            ha="center",
            va="center",
            rotation=90 if rotate else 0,
            fontsize=10,
            clip_on=True,
        )

    ax.set_title(
        f"{y_axis} by {x_axis}: "
        f"Top {top_n} (out of {distinct_products}) covers {coverage_pct}% of the result",
        fontsize=16
    )

    ax.axis("off")
    fig.tight_layout()
    return fig, coverage_pct


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_numeric_dtype(df[c])]


def _category_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_categorical_dtype(df[c])]


def _text_columns(df: pd.DataFrame) -> list[str]:
    """
    Text fields (kept as-is) that can still be used as labels in charts.
    """
    cols: list[str] = []
    for c in df.columns:
        s = df[c]
        if is_categorical_dtype(s) or is_numeric_dtype(s) or is_datetime64_any_dtype(s):
            continue
        if s.dtype == "object" or str(s.dtype) == "string":
            cols.append(c)
    return cols


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


def _label_breakdown_option(col: str, *, text_cols: list[str]) -> str:
    return col


def _is_text_field(col: str, *, text_cols: list[str]) -> bool:
    return col in text_cols


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

    if "viz_has_generated" not in st.session_state:
        st.session_state.viz_has_generated = False

    info_slot = st.empty()
    if not st.session_state.viz_has_generated:
        info_slot.info("Choose what to measure and how to break it down, then generate a chart.")

    numeric_cols = _numeric_columns(df)
    cat_cols = _category_columns(df)
    text_cols = _text_columns(df)
    date_cols = _date_columns(df, column_semantics, sem_date_value)

    if not numeric_cols:
        st.warning("No numeric fields available to measure.")
        st.stop()

    mode = st.radio(
        "Which kind analysis you want to perform?",
        options=[
            "Compare categories",
            "Trend over time",
            "Relationship between two metrics",
        ],
        horizontal=True,
        help=(
            "1. **Compare categories**\n"
            "Compare totals or averages across groups or categories.\n"
            "2. **Trend over time**\n"
            "See how a metric changes over time, including trends, peaks, and drops.\n"
            "3. **Relationship between two metrics**\n"
            "See if two metrics move together and identify patterns or outliers."
        ),
    )

    if mode == "Compare categories":
        st.caption("Compare totals/averages across groups or categories.")
    elif mode == "Trend over time":
        st.caption("See how a metric changes over time: trends, peaks, and drops.")
    else:
        st.caption("See if two metrics move together and identify patterns or outliers.")

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
        generate_clicked = st.button("Generate Chart") if ready else False

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
        compare_options = cat_cols + [c for c in text_cols if c not in cat_cols]
        if not compare_options:
            st.warning("No category or text fields available to compare.")
            st.stop()

        x_axis = st.selectbox(
            "Group by (Category)",
            options=compare_options,
            index=_auto_index_for_single_option(compare_options),
            placeholder="Choose a field",
            format_func=lambda c: _label_breakdown_option(c, text_cols=text_cols),
            help="How do you want to break the measure down (x-axis).",
            key=x_key,
        )

    else:
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

    heatmap_by: Optional[str] = None
    HEAT_TOP_ROWS = 20
    HEAT_TOP_COLS = 20

    # For dynamic Treemap label text in "Visual type"
    treemap_coverage_pct: Optional[int] = None

    if ready:
        y_is_numeric = is_numeric_dtype(df[y_axis])

        x_s = df[x_axis]
        x_sem = column_semantics.get(x_axis)

        x_is_date_like = (x_sem == sem_date_value) or is_datetime64_any_dtype(x_s)
        x_is_cat = is_categorical_dtype(x_s)
        x_is_num = is_numeric_dtype(x_s)
        x_is_text = (x_s.dtype == "string") or (x_s.dtype == "object")

        # can_heatmap/can_treemap only for Compare categories + text breakdown
        heatmap_options = [c for c in (cat_cols + text_cols) if c != x_axis]
        can_heatmap = (mode == "Compare categories") and x_is_text and (len(heatmap_options) > 0)
        can_treemap = (mode == "Compare categories") and x_is_text

        # Precompute coverage_pct for dynamic Treemap label (lightweight groupby)
        if can_treemap and y_is_numeric:
            try:
                _, treemap_coverage_pct = make_treemap_pareto_text_breakdown(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    agg_func=agg_func,
                    agg_sum_value=agg_sum_value,
                    agg_avg_value=agg_avg_value,
                )
                plt.close("all")
            except Exception:
                treemap_coverage_pct = None

        plot_list = compatible_plots(
            x_is_categorical=x_is_cat,
            x_is_text=x_is_text,
            x_is_date_like=x_is_date_like,
            x_is_numeric=x_is_num,
            y_is_numeric=y_is_numeric,
            agg_func=agg_func,
            agg_sum_value=agg_sum_value,
            can_heatmap=can_heatmap,
            can_treemap=can_treemap,
        )

        if not plot_list:
            st.warning("No compatible charts for the selected fields.")
            return

        # Local plot labels (so UI text can depend on the selected x-axis type)
        plot_labels_ui = dict(PLOT_LABELS)
        if x_is_text:
            plot_labels_ui[PLOT_TOPN_TABLE] = "Top categories (table, limited to Top 20)"

        # Dynamic treemap label (includes X = coverage_pct)
        if can_treemap:
            x_pct = treemap_coverage_pct if treemap_coverage_pct is not None else 0
            plot_labels_ui[PLOT_TREEMAP] = f"Treemap limited to the groups driving {x_pct}% of the result"

        if len(plot_list) == 1:
            plot_type = plot_list[0]
        else:
            plot_type = st.selectbox(
                "Visual type",
                options=plot_list,
                index=_auto_index_for_single_option(plot_list),
                format_func=lambda p: plot_labels_ui.get(p, p),
                placeholder="Choose",
                key=plot_key,
            )

        if plot_type == PLOT_HEATMAP:
            heatmap_by = st.selectbox(
                "Second category (for heatmap)",
                options=heatmap_options,
                index=_auto_index_for_single_option(heatmap_options),
                placeholder="Choose a field",
                help="Adds a second category so the heatmap can show a grid of values.",
                key="heatmap_by",
            )

    ready_to_generate = ready and (plot_type is not None)
    if plot_type == PLOT_HEATMAP:
        ready_to_generate = ready_to_generate and (heatmap_by is not None)

    generate_clicked = st.button("Generate") if ready_to_generate else False

    if generate_clicked:
        st.session_state.viz_has_generated = True
        info_slot.empty()

        agg_part = f" ({agg_func})" if agg_func else ""

        # caption title uses the same UI label mapping where possible
        caption_label = PLOT_LABELS.get(plot_type, plot_type)
        if plot_type == PLOT_TREEMAP:
            x_pct = treemap_coverage_pct if treemap_coverage_pct is not None else 0
            caption_label = f"Treemap limited to the groups driving {x_pct}% of the result"

        title = f"{caption_label}: {y_axis}{agg_part} by {x_axis}"

        if plot_type == PLOT_HEATMAP and heatmap_by is not None:
            title += f" and {heatmap_by}"

        st.caption(title)

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
                st.pyplot(fig)
                plt.close(fig)

            elif plot_type == PLOT_TOPN_TABLE:
                summary_df = make_category_summary_table(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    agg_func=agg_func,
                    agg_sum_value=agg_sum_value,
                    agg_avg_value=agg_avg_value,
                    top_rows=20,
                )
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            elif plot_type == PLOT_PIE:
                fig = make_pie_chart(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    agg_func=agg_func,
                    agg_sum_value=agg_sum_value,
                )
                st.pyplot(fig)
                plt.close(fig)

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
                st.pyplot(fig)
                plt.close(fig)

            elif plot_type == PLOT_CUMULATIVE_LINE:
                fig = make_cumulative_line_chart(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    x_semantic=x_sem,
                    sem_date_value=sem_date_value,
                    to_datetime_fn=to_datetime_fn,
                )
                st.pyplot(fig)
                plt.close(fig)

            elif plot_type == PLOT_HEATMAP:
                fig = make_heatmap(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    heatmap_by=str(heatmap_by),
                    agg_func=agg_func,
                    agg_sum_value=agg_sum_value,
                    agg_avg_value=agg_avg_value,
                    top_rows=HEAT_TOP_ROWS,
                    top_cols=HEAT_TOP_COLS,
                )
                st.pyplot(fig)
                plt.close(fig)

            elif plot_type == PLOT_TREEMAP:
                fig, treemap_coverage_pct = make_treemap_pareto_text_breakdown(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    agg_func=agg_func,
                    agg_sum_value=agg_sum_value,
                    agg_avg_value=agg_avg_value,
                )
                st.pyplot(fig)
                plt.close(fig)

            else:
                st.error(f"Unsupported plot type: {plot_type}")
                st.stop()

        except Exception as e:
            st.error(f"❌ Failed to generate output: {e}")
            st.exception(e)
