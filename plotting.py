# plotting.py
from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas.api.types import is_categorical_dtype, is_datetime64_any_dtype, is_numeric_dtype


# Plot names
PLOT_BAR = "Bar Chart"
PLOT_PIE = "Pie Chart"
PLOT_LINE = "Line Chart"


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

    if x_is_categorical and y_is_numeric:
        plots.append(PLOT_BAR)
        if agg_func == agg_sum_value:
            plots.append(PLOT_PIE)

    if x_is_date_like and y_is_numeric:
        plots.append(PLOT_LINE)

    if x_is_numeric and y_is_numeric:
        plots.append(PLOT_LINE)

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

    sns.barplot(x=bar_df[x_axis], y=bar_df[value_col], ax=ax)
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
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    pie_data = df.groupby(x_axis)[y_axis].sum().sort_values(ascending=False)

    # Avoid clutter for tiny slices
    def autopct_fn(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= 3 else ""

    wedges, texts, autotexts = ax.pie(
        pie_data.values,
        labels=None,                 # ✅ put labels in legend instead
        autopct=autopct_fn,          # ✅ hide tiny percentages
        startangle=90,
        counterclock=False,
        pctdistance=0.7,             # ✅ keep percentages inside
        textprops={"fontsize": 10},
        colors=sns.color_palette("pastel", n_colors=len(pie_data)),
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )

    ax.axis("equal")
    ax.set_title(f"{PLOT_PIE} of {y_axis} by {x_axis}", fontsize=12)

    # ✅ legend for category names (no collisions)
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
    fig, ax = plt.subplots(figsize=(6, 4))

    line_df = df[[x_axis, y_axis]].dropna().copy()

    if x_is_date_like:
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
        line_df = line_df.sort_values(by=x_axis)
        sns.lineplot(x=line_df[x_axis], y=line_df[y_axis], ax=ax)

    ax.tick_params(axis="y", labelsize=10)
    ax.set_title(f"{PLOT_LINE} of {y_axis} by {x_axis}", fontsize=12)
    ax.set_xlabel(x_axis, fontsize=10)
    ax.set_ylabel(y_axis, fontsize=10)

    fig.tight_layout()
    return fig


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
        info_slot.info("Select the fields and type of plot you want to generate.")

    x_axis = st.selectbox(
        "Select the X-axis",
        options=columns,
        index=None,
        placeholder="Choose a column",
        key=x_key,
    )
    if x_axis is None:
        st.stop()

    x_s = df[x_axis]
    x_sem = column_semantics.get(x_axis)

    x_is_date_like = (x_sem == sem_date_value) or is_datetime64_any_dtype(x_s)
    x_is_cat = is_categorical_dtype(x_s)
    x_is_numeric = is_numeric_dtype(x_s)

    agg_func: Optional[str] = None

    if x_is_cat or x_is_date_like:
        col_y, col_agg = st.columns([2, 1])

        with col_y:
            y_axis = st.selectbox(
                "Select the Y-axis",
                options=columns,
                index=None,
                placeholder="Choose a column",
                key=y_key,
            )

        with col_agg:
            agg_func = st.selectbox(
                "Aggregation",
                options=agg_options,
                index=None,
                placeholder="Choose",
                key=agg_key,
            )
    else:
        y_axis = st.selectbox(
            "Select the Y-axis",
            options=columns,
            index=None,
            placeholder="Choose a column",
            key=y_key,
        )

    if y_axis is None:
        st.stop()

    if (x_is_cat or x_is_date_like) and agg_func is None:
        st.stop()

    y_s = df[y_axis]
    y_is_numeric = is_numeric_dtype(y_s)

    plot_list = compatible_plots(
        x_is_categorical=x_is_cat,
        x_is_date_like=x_is_date_like,
        x_is_numeric=x_is_numeric,
        y_is_numeric=y_is_numeric,
        agg_func=agg_func,
        agg_sum_value=agg_sum_value,
    )

    if not plot_list:
        st.warning("No compatible plots for the selected columns.")
        st.stop()

    if len(plot_list) == 1:
        plot_type = plot_list[0]
    else:
        plot_type = st.selectbox(
            "Select the type of plot",
            options=plot_list,
            index=None,
            placeholder="Choose a plot type",
            key=plot_key,
        )
        if plot_type is None:
            st.stop()

    generate_clicked = st.button("Generate Plot")

    if generate_clicked:
        st.session_state.viz_has_generated = True
        info_slot.empty()

        agg_part = f" ({agg_func})" if agg_func else ""
        st.caption(f"{plot_type} of{agg_part}  {y_axis} by {x_axis}")

        try:
            if plot_type == PLOT_BAR:
                fig = make_bar_chart(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    agg_func=agg_func,  # required here
                    agg_sum_value=agg_sum_value,
                    agg_avg_value=agg_avg_value,
                )

            elif plot_type == PLOT_PIE:
                fig = make_pie_chart(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                )

            elif plot_type == PLOT_LINE:
                if not y_is_numeric:
                    st.error("Line Chart requires a numeric Y-axis.")
                    st.stop()

                fig = make_line_chart(
                    df,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    x_is_date_like=x_is_date_like,
                    agg_func=agg_func,  # only required when x_is_date_like
                    x_semantic=x_sem,
                    sem_date_value=sem_date_value,
                    agg_sum_value=agg_sum_value,
                    agg_avg_value=agg_avg_value,
                    to_datetime_fn=to_datetime_fn,
                )

            else:
                st.error(f"Unsupported plot type: {plot_type}")
                st.stop()

            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"❌ Failed to generate plot: {e}")
            st.exception(e)
