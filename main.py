import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotting

from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_categorical_dtype,
    is_object_dtype,
)

# Set the page config
st.set_page_config(page_title='Data Visualizer',
                   layout='centered',
                   page_icon='📊')

st.title('📈  Data Visualizer')

##################################################################################
# selecting the file

working_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = f"{working_dir}/data"

# Ensure the data folder exists
if not os.path.isdir(folder_path):
    st.error(f"Data folder not found: {folder_path}")
    st.stop()

files = [f for f in os.listdir(folder_path) if f.endswith(('.csv', '.xlsx'))]

# No files found
if not files:
    st.warning("No .csv or .xlsx files found in the data folder.")
    st.stop()

selected_file = st.selectbox('Select a file', files, index=None)

if selected_file:
    file_path = os.path.join(folder_path, selected_file)

    # --- persist df in session state ---
    if "df" not in st.session_state:
        st.session_state.df = None
    if "loaded_file" not in st.session_state:
        st.session_state.loaded_file = None
    if "loaded_sheet" not in st.session_state:
        st.session_state.loaded_sheet = None

    # ✅ metadata ONLY for semantic overrides like "date"
    if "column_semantics" not in st.session_state:
        st.session_state.column_semantics = {}

    def _reset_column_semantics():
        # reset on new dataset to avoid carrying settings to unrelated files
        st.session_state.column_semantics = {}

    #  Loading Excel using Context Manager: close file at end of the run/ no zombie file handles
    if selected_file.endswith('.xlsx'):
        try:
            with pd.ExcelFile(file_path) as xls:
                sheet_name = st.selectbox('Select a sheet', xls.sheet_names, index=None)

                if sheet_name is None:
                    st.stop()

                else:
                    is_new_dataset = (
                        st.session_state.df is None
                        or st.session_state.loaded_file != selected_file
                        or st.session_state.loaded_sheet != sheet_name
                    )
                    if is_new_dataset:
                        st.session_state.df = pd.read_excel(xls, sheet_name=sheet_name)
                        st.session_state.loaded_file = selected_file
                        st.session_state.loaded_sheet = sheet_name
                        _reset_column_semantics()

        except Exception as e:
            st.error("❌ Failed to read the Excel file. Please check the file and selected sheet.")
            st.exception(e)
            st.stop()

    else:
        try:
            is_new_dataset = (
                st.session_state.df is None
                or st.session_state.loaded_file != selected_file
                or st.session_state.loaded_sheet is not None
            )
            if is_new_dataset:
                st.session_state.df = pd.read_csv(file_path)
                st.session_state.loaded_file = selected_file
                st.session_state.loaded_sheet = None
                _reset_column_semantics()

        except Exception as e:
            st.error("❌ Failed to read the CSV file. Please ensure it is a valid CSV.")
            st.exception(e)
            st.stop()


    if st.session_state.df is None:
        st.error("No data loaded. Please select a valid file/sheet.")
        st.stop()

    df = st.session_state.df
    columns = df.columns.tolist()

    st.write("")

##################################################################################
# Data Preview (placeholder so it updates in-place)
    st.header("🔍 Data Preview")

    def highlight_first_row(s):
        return ["background-color: lightyellow" if s.name == "Type" else "" for _ in s]

    def render_preview(current_df: pd.DataFrame):
        head_df_local = current_df.head().copy()

        # ✅ display formatting: show date-only without "00:00:00"
        for c in head_df_local.columns:
            if st.session_state.column_semantics.get(c) == "date":
                s = pd.to_datetime(head_df_local[c], errors="coerce")
                head_df_local[c] = s.dt.strftime("%Y-%m-%d")

        # Type row: pandas dtype except if semantic "date"
        display_types = []
        for c in head_df_local.columns:
            if st.session_state.column_semantics.get(c) == "date":
                display_types.append("date")
            else:
                display_types.append(str(current_df[c].dtype))

        dtypes_row_local = pd.Series(display_types, index=head_df_local.columns, name="Type")
        combined_df_local = pd.concat([pd.DataFrame(dtypes_row_local).T, head_df_local])
        return combined_df_local.style.apply(highlight_first_row, axis=1)

    preview_placeholder = st.empty()
    preview_placeholder.dataframe(render_preview(df))

##################################################################################
# Changing the Type (safe converters only)  [BOOL REMOVED]

    st.header('🔧 Changing the Type')
    st.info("Change the data type of columns if they were not interpreted correctly.")

    # Show confirmation message if present
    if "change_type_success_msg" in st.session_state:
        st.success(st.session_state["change_type_success_msg"])

    # Reset widget selections on the next run (before widgets are created)
    if st.session_state.get("_reset_change_type_widgets", False):
        st.session_state["change_type_cols"] = []
        st.session_state["change_type_target"] = None
        st.session_state["_reset_change_type_widgets"] = False

    col1, col2, col3 = st.columns([3, 3, 1.5])

    with col1:
        choose_cols = st.multiselect(
            'Change the type of specific columns',
            options=columns,
            key="change_type_cols"
        )

    with col2:
        possible_types = ['int64', 'float64', 'datetime', 'date', 'category']
        choose_type = st.selectbox(
            'Set type to:',
            options=possible_types,
            index=None,
            placeholder="Choose an option",
            key="change_type_target"
        )

    with col3:
        st.markdown("<div style='margin-bottom: 6px; font-weight: bold;'>Confirm</div>", unsafe_allow_html=True)
        change_type_clicked = st.button("Change Type")

    if choose_cols and choose_type and change_type_clicked:
        try:
            for col in choose_cols:
                s = st.session_state.df[col]

                if choose_type == 'category':
                    st.session_state.df[col] = s.astype('category')
                    st.session_state.column_semantics.pop(col, None)

                elif choose_type == 'datetime':
                    st.session_state.df[col] = pd.to_datetime(s, errors='coerce')
                    st.session_state.column_semantics.pop(col, None)

                elif choose_type == 'date':
                    dt = pd.to_datetime(s, errors='coerce')
                    st.session_state.df[col] = dt.dt.normalize()
                    st.session_state.column_semantics[col] = "date"

                elif choose_type == 'float64':
                    st.session_state.df[col] = pd.to_numeric(s, errors="coerce")
                    st.session_state.column_semantics.pop(col, None)

                elif choose_type == 'int64':
                    num = pd.to_numeric(s, errors="coerce")
                    st.session_state.df[col] = num.astype("Int64")
                    st.session_state.column_semantics.pop(col, None)

            df = st.session_state.df

            # Persist confirmation text across reruns
            st.session_state["change_type_success_msg"] = (
                f"✅ Column(s) '{', '.join(choose_cols)}' type successfully changed to '{choose_type}'. See data preview above"
            )

            preview_placeholder.dataframe(render_preview(df))

            # Trigger widget reset on next run
            st.session_state["_reset_change_type_widgets"] = True
            st.rerun()

        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")



##################################################################################
# --- KPI SECTION ---

    st.header('🔢 Key Performance Indicators (KPIs)')

    # info box
    st.info("Select a data field below to view its key statistics.")

    # Let the user choose a column; start with no preselected value
    kpi_column = st.selectbox(
        'Select the field for KPI calculation',
        options=columns,
        index=None,
        key='kpi_field_auto'
    )

    # Only compute KPIs after a column is selected
    if kpi_column is not None:
        s0 = df[kpi_column]
        semantic = st.session_state.column_semantics.get(kpi_column)

        # Numeric KPIs
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

            col1, col2, col3 = st.columns(3)
            for i in range(3):
                title, result = kpi_list[i]

                # ✅ no commas for counts
                if title in ("Count (valid)", "Missing"):
                    display_value = str(int(result))
                elif isinstance(result, (float, int)):
                    display_value = f"{result:,.2f}"
                else:
                    display_value = str(result)

                with [col1, col2, col3][i]:
                    st.metric(label=f"{title} of {kpi_column}", value=display_value)

            col4, col5, col6 = st.columns(3)
            for i in range(3):
                title, result = kpi_list[i + 3]

                if title in ("Count (valid)", "Missing"):
                    display_value = str(int(result))
                elif isinstance(result, (float, int)):
                    display_value = f"{result:,.2f}"
                else:
                    display_value = str(result)

                with [col4, col5, col6][i]:
                    st.metric(label=f"{title} of {kpi_column}", value=display_value)

        # DATE / DATETIME (semantic "date" OR real datetime dtype OR parseable object)
        elif semantic == "date" or is_datetime64_any_dtype(s0) or (
            is_object_dtype(s0) and pd.to_datetime(s0, errors="coerce").notna().mean() > 0.6
        ):
            s = pd.to_datetime(s0, errors="coerce")
            valid = s.dropna()
            missing_count = int(s.isna().sum())

            if valid.empty:
                st.warning(f"Column '{kpi_column}' looks like a date/datetime field, but no valid dates could be parsed.")
            else:
                min_dt = valid.min()
                max_dt = valid.max()
                span_days = int((max_dt - min_dt).days)
                unique_dates = int(valid.dt.date.nunique())

                min_disp = min_dt.date() if semantic == "date" else min_dt
                max_disp = max_dt.date() if semantic == "date" else max_dt

                kpi_calculations = {
                    "Count (valid)": int(valid.shape[0]),
                    "Missing": missing_count,
                    "Unique": unique_dates,
                    "Min": min_disp,
                    "Max": max_disp,
                    "Span (days)": span_days,
                }

                kpi_list = list(kpi_calculations.items())

                col1, col2, col3 = st.columns(3)
                for i in range(3):
                    title, result = kpi_list[i]
                    with [col1, col2, col3][i]:
                        st.metric(label=f"{title} of {kpi_column}", value=str(result))

                col4, col5, col6 = st.columns(3)
                for i in range(3):
                    title, result = kpi_list[i + 3]
                    with [col4, col5, col6][i]:
                        st.metric(label=f"{title} of {kpi_column}", value=str(result))

        # CATEGORY (only 3 measures)
        elif is_categorical_dtype(s0):
            s = s0
            kpi_calculations = {
                "Count (valid)": int(s.notna().sum()),
                "Missing": int(s.isna().sum()),
                "Unique categories": int(s.nunique(dropna=True)),
            }

            kpi_list = list(kpi_calculations.items())
            col1, col2, col3 = st.columns(3)
            for i in range(3):
                title, result = kpi_list[i]
                with [col1, col2, col3][i]:
                    st.metric(label=f"{title} of {kpi_column}", value=str(result))

        else:
            st.warning(f"The column '{kpi_column}' is not a supported type for KPIs (dtype={df[kpi_column].dtype}).")

##################################################################################
# Charts & Visuals

    st.header('📊 Charts & Visuals')

    x_axis = st.selectbox('Select the X-axis', options=columns)
    y_axis = st.selectbox('Select the Y-axis', options=columns)

    aggregation_functions = ["sum", "average"]
    agg_func = None           # operation to apply
    value_col_name = None     # name of aggregated result column
    plot_list = []

    x_s = df[x_axis]
    y_s = df[y_axis]

    x_sem = st.session_state.column_semantics.get(x_axis)

    if is_categorical_dtype(x_s) and is_numeric_dtype(y_s):
        agg_func = st.selectbox('Select the aggregation function', options=aggregation_functions)
        plot_list.append('Bar Chart')
        if agg_func == "sum":
            plot_list.append('Pie Chart')

    elif is_numeric_dtype(x_s) and is_numeric_dtype(y_s):
        plot_list.extend(['Scatter Plot', 'Line Chart'])

    elif (x_sem == "date" or is_datetime64_any_dtype(x_s)) and is_numeric_dtype(y_s):
        plot_list.append('Line Chart')

    # fallback: no compatible plots
    if not plot_list:
        st.warning("No compatible plots for the selected columns.")
        st.stop()


    plot_type = st.selectbox('Select the type of plot', options=plot_list)

    if st.button('Generate Plot'):
        fig, ax = plt.subplots(figsize=(6, 4))

        if plot_type == 'Bar Chart':

            # safety check
            if agg_func is None:
                st.error("Please choose an aggregation function before generating the bar chart.")
                st.stop()

            if agg_func == "sum":
                bar_df = df.groupby(x_axis)[y_axis].sum().reset_index(name="sum")
                value_col_name = "sum"
            elif agg_func == "average":
                bar_df = df.groupby(x_axis)[y_axis].mean().reset_index(name="average")
                value_col_name = "average"
            else:
                st.error(f"Unsupported aggregation: {agg_func}")
                st.stop()

            bar_df = bar_df.sort_values(by=value_col_name, ascending=False)

            sns.barplot(x=bar_df[x_axis], y=bar_df[value_col_name], ax=ax)
            ax.tick_params(axis='x', rotation=90)

            # desgin/ layout: leave space above bars for labels
            ymax = bar_df[value_col_name].max()
            if pd.notna(ymax) and ymax != 0:
                ax.set_ylim(top=ymax * 1.10)
    
            for p in ax.patches:
                height = p.get_height()
                if pd.isna(height):
                    continue

                if agg_func == "average":
                    label = f"{height:.2f}"
                else:  # sum
                    if float(height).is_integer():
                        label = f"{int(height)}"
                    else:
                        label = f"{height:.2f}"

                ax.annotate(
                    label,
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    color='black',
                    xytext=(0, 3),
                    textcoords='offset points'
                )

        elif plot_type == "Pie Chart":
            pie_data = df.groupby(x_axis)[y_axis].sum()
            ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90,
                   colors=sns.color_palette("pastel"))

        elif plot_type == 'Line Chart':
            #Guard
            if not is_numeric_dtype(df[y_axis]):
                st.error("Line Chart requires a numeric Y-axis.")
                st.stop()
            # handle depending on x-axis type       
            if st.session_state.column_semantics.get(x_axis) == "date" or is_datetime64_any_dtype(df[x_axis]):
                plotting.create_lineChart_Date(df, x_axis, y_axis, ax)
            else:
                sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)

        elif plot_type == 'Scatter Plot':
            sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)

        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        plt.title(f'{plot_type} of {y_axis} vs {x_axis}', fontsize=12)
        plt.xlabel(x_axis, fontsize=10)
        plt.ylabel(y_axis, fontsize=10)

        st.pyplot(fig)
