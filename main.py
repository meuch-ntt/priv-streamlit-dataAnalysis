import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.dates as mdates
import plotting
import openpyxl

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

files = [f for f in os.listdir(folder_path) if f.endswith(('.csv', '.xlsx'))]
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
    if "col_semantic_types" not in st.session_state:
        st.session_state.col_semantic_types = {}

    def _reset_semantic_types():
        # reset on new dataset to avoid carrying settings to unrelated files
        st.session_state.col_semantic_types = {}

    if selected_file.endswith('.xlsx'):
        xls = pd.ExcelFile(file_path)
        sheet_name = st.selectbox('Select a sheet', xls.sheet_names)

        if sheet_name:
            is_new_dataset = (
                st.session_state.df is None
                or st.session_state.loaded_file != selected_file
                or st.session_state.loaded_sheet != sheet_name
            )
            if is_new_dataset:
                st.session_state.df = pd.read_excel(xls, sheet_name=sheet_name)
                st.session_state.loaded_file = selected_file
                st.session_state.loaded_sheet = sheet_name
                _reset_semantic_types()
        else:
            st.stop()

    else:
        is_new_dataset = (
            st.session_state.df is None
            or st.session_state.loaded_file != selected_file
            or st.session_state.loaded_sheet is not None
        )
        if is_new_dataset:
            st.session_state.df = pd.read_csv(file_path)
            st.session_state.loaded_file = selected_file
            st.session_state.loaded_sheet = None
            _reset_semantic_types()

    df = st.session_state.df
    columns = df.columns.tolist()

    st.write("")



##################################################################################
# Data Preview (placeholder so it updates in-place)
    st.markdown('<div id="data-preview"></div>', unsafe_allow_html=True)
    st.header("🔍 Data Preview")

    def highlight_first_row(s):
        return ["background-color: lightyellow" if s.name == "Type" else "" for _ in s]

    def render_preview(current_df: pd.DataFrame):
        head_df_local = current_df.head().copy()

        # ✅ display formatting: show date-only without "00:00:00"
        for c in head_df_local.columns:
            if st.session_state.col_semantic_types.get(c) == "date":
                s = pd.to_datetime(head_df_local[c], errors="coerce")
                head_df_local[c] = s.dt.strftime("%Y-%m-%d")

        # Type row: pandas dtype except if semantic "date"
        display_types = []
        for c in head_df_local.columns:
            if st.session_state.col_semantic_types.get(c) == "date":
                display_types.append("date")
            else:
                display_types.append(str(current_df[c].dtype))

        dtypes_row_local = pd.Series(display_types, index=head_df_local.columns, name="Type")
        combined_df_local = pd.concat([pd.DataFrame(dtypes_row_local).T, head_df_local])
        return combined_df_local.style.apply(highlight_first_row, axis=1)

    preview_placeholder = st.empty()
    preview_placeholder.dataframe(render_preview(df))

##################################################################################
# Changing the Type (no extra table here)

    st.header('🔧 Changing the Type')

    if 'lastState' not in st.session_state:
        st.session_state.lastState = None

    if st.session_state.lastState is None:
        st.warning("⚠️ In order to make changes: select the column(s) and desired type then confirm ")

    col1, col2, col3 = st.columns([3, 3, 1.5])

    with col1:
        choose_cols = st.multiselect('Change the type of specific columns', options=columns)

    with col2:
        possible_types = ['int64', 'float64', 'bool', 'datetime', 'date', 'category']
        choose_type = st.selectbox(
            'Set type to:',
            options=possible_types,
            index=None,
            placeholder="Choose an option"
        )


    with col3:
        st.markdown("<div style='margin-bottom: 6px; font-weight: bold;'>Confirm</div>", unsafe_allow_html=True)
        change_type_clicked = st.button("Change Type")

    if choose_cols and choose_type and change_type_clicked:
        try:
            for col in choose_cols:
                if choose_type == 'category':
                    st.session_state.df[col] = st.session_state.df[col].astype('category')
                    # category is a real dtype; no semantic override needed
                    st.session_state.col_semantic_types.pop(col, None)

                elif choose_type == 'datetime':
                    st.session_state.df[col] = pd.to_datetime(st.session_state.df[col], errors='coerce')
                    st.session_state.col_semantic_types.pop(col, None)

                elif choose_type == 'date':
                    dt = pd.to_datetime(st.session_state.df[col], errors='coerce')
                    st.session_state.df[col] = dt.dt.normalize()  # keep datetime dtype, but date semantics
                    st.session_state.col_semantic_types[col] = "date"

                else:
                    st.session_state.df[col] = st.session_state.df[col].astype(choose_type)
                    st.session_state.col_semantic_types.pop(col, None)

            df = st.session_state.df

            st.success(f"✅ Column(s) '{', '.join(choose_cols)}' type successfully changed to '{choose_type}'.")

            # ✅ Update Data Preview in-place (no new table below)
            preview_placeholder.dataframe(render_preview(df))
            st.session_state.lastState = True

        except ValueError as e:
            st.error(f"❌ Error changing type: {e}")
        except TypeError as e:
            st.error(f"❌ Error changing type: {e}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")

##################################################################################
# --- KPI SECTION ---

    st.header('🔢 Key Performance Indicators (KPIs)')

    select_options = ["--- Select a Field ---"] + columns
    kpi_column = st.selectbox('Select the field for KPI calculation',
                              options=select_options,
                              key='kpi_field_auto')

    if kpi_column == "--- Select a Field ---":
        st.info("Please select a data field above to view its key statistics.")

    elif kpi_column in columns:
        s0 = df[kpi_column]
        semantic = st.session_state.col_semantic_types.get(kpi_column)

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

            col1, col2, col3 = st.columns(3)
            for i in range(3):
                title, result = kpi_list[i]
                display_value = f"{result:,.2f}" if isinstance(result, float) else str(result)
                with [col1, col2, col3][i]:
                    st.metric(label=f"{title} of {kpi_column}", value=display_value)

            col4, col5, col6 = st.columns(3)
            for i in range(3):
                title, result = kpi_list[i + 3]
                display_value = f"{result:,.2f}" if isinstance(result, (float, int)) else str(result)
                with [col4, col5, col6][i]:
                    st.metric(label=f"{title} of {kpi_column}", value=display_value)

        # DATE / DATETIME (semantic "date" OR real datetime dtype OR parseable object)
        elif semantic == "date" or is_datetime64_any_dtype(s0) or is_object_dtype(s0):
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

                # If semantic date, show dates only, else show timestamps
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

        # CATEGORY
        elif is_categorical_dtype(s0):
            s = s0
            kpi_calculations = {
                "Count (valid)": int(s.notna().sum()),
                "Missing": int(s.isna().sum()),
                "Unique": int(s.nunique(dropna=True)),
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

    x_axis = st.selectbox('Select the X-axis', options=columns + ["None"])
    y_axis = st.selectbox('Select the Y-axis', options=columns + ["None"])

    aggregation_functions = ["sum", "average", "count"]
    z_axis = "None"
    plot_list = []

    if x_axis != "None" and y_axis != "None":
        x_s = df[x_axis]
        y_s = df[y_axis]

        x_sem = st.session_state.col_semantic_types.get(x_axis)

        if is_categorical_dtype(x_s) and is_numeric_dtype(y_s):
            plot_list.extend(['Bar Chart', 'Pie Chart', 'Box Chart'])

        elif is_numeric_dtype(x_s) and is_numeric_dtype(y_s):
            plot_list.extend(['Scatter Plot', 'Line Chart'])

        elif x_sem == "date" or is_datetime64_any_dtype(x_s):
            plot_list.append('Line Chart')

        elif is_categorical_dtype(x_s) and is_object_dtype(y_s):
            z_axis = st.selectbox('Select the aggregation funtion', options=aggregation_functions + ["None"])
            plot_list.append('Bar Chart')

    if x_axis == "None" or y_axis == "None":
        plot_list = ['Line Chart', 'Scatter Plot', 'Distribution Plot', 'Count Plot']

    plot_type = st.selectbox('Select the type of plot', options=plot_list)

    if st.button('Generate Plot'):
        fig, ax = plt.subplots(figsize=(6, 4))

        if plot_type == 'Bar Chart' and z_axis == "count":
            count_df = df.groupby(x_axis).size().reset_index(name='count')
            count_df = count_df.sort_values(by='count', ascending=False)
            sns.barplot(x=count_df[x_axis], y=count_df['count'], ax=ax)
            ax.tick_params(axis='x', rotation=90)

            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=12, color='black',
                            xytext=(0, 5), textcoords='offset points')

        elif plot_type == 'Bar Chart' and z_axis != "count":
            sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax)

        elif plot_type == 'Box Chart':
            # sns.boxplot(x=df[x_axis], y=df[y_axis], ax=ax)
            pass

        elif plot_type == "Pie Chart":
            pie_data = df.groupby(x_axis)[y_axis].sum()
            ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90,
                   colors=sns.color_palette("pastel"))

        elif plot_type == 'Line Chart':
            if st.session_state.col_semantic_types.get(x_axis) == "date" or is_datetime64_any_dtype(df[x_axis]):
                plotting.create_lineChart_Date(df, x_axis, y_axis, ax)
            else:
                sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)

        elif plot_type == 'Scatter Plot':
            sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)

        elif plot_type == 'Distribution Plot':
            sns.histplot(df[x_axis], kde=True, ax=ax)
            y_axis = 'Density'

        elif plot_type == 'Count Plot':
            sns.countplot(x=df[x_axis], ax=ax)
            y_axis = 'Count'

        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        plt.title(f'{plot_type} of {y_axis} vs {x_axis}', fontsize=12)
        plt.xlabel(x_axis, fontsize=10)
        plt.ylabel(y_axis, fontsize=10)

        st.pyplot(fig)
