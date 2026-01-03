import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import pandas as pd




def bar_count(df, x_axis, plot_type, aggregation_functions):
    a=5



def create_lineChart_Date(df,x_axis, y_axis, ax):
    sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)

    # Format the x-axis dates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

