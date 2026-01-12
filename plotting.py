# plotting.py
from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Plot names
PLOT_BAR = "Bar Chart"
PLOT_PIE = "Pie Chart"
PLOT_LINE = "Line Chart"


def compatible_plots(
    *,
    x_is_categorical: bool,
    x_is_date_like: bool,
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
    fig, ax = plt.subplots(figsize=(6, 4))

    pie_data = df.groupby(x_axis)[y_axis].sum()

    ax.pie(
        pie_data,
        labels=pie_data.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("pastel"),
    )
    ax.axis("equal")
    ax.set_title(f"{PLOT_PIE} of {y_axis} by {x_axis}", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.tight_layout()
    return fig


def make_line_chart(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_axis: str,
    agg_func: str,
    x_semantic: Optional[str],
    sem_date_value: str,
    agg_sum_value: str,
    agg_avg_value: str,
    to_datetime_fn: Callable[[pd.Series], pd.Series],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))

    line_df = df[[x_axis, y_axis]].dropna().copy()
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
    ax.tick_params(axis="y", labelsize=10)

    ax.set_title(f"{PLOT_LINE} of {y_axis} by {x_axis}", fontsize=12)
    ax.set_xlabel(x_axis, fontsize=10)
    ax.set_ylabel(y_axis, fontsize=10)

    fig.tight_layout()
    return fig
