import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.dates as mdates
import plotting
import openpyxl



# Set the page config
st.set_page_config(page_title='Data Visualizer',
                   layout='centered',
                   page_icon='📊')

# Title
st.title('📈  Data Visualizer')

##################################################################################
# selecting the fil

working_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the folder where your CSV files are located
folder_path = f"{working_dir}/data"  # Update this to your folder path

# List all files in the folder
##files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
files = [f for f in os.listdir(folder_path) if f.endswith(('.csv', '.xlsx'))]

# Dropdown to select a file
selected_file = st.selectbox('Select a file', files, index=None)

if selected_file:
    # Construct the full path to the file
    file_path = os.path.join(folder_path, selected_file)
    
    if selected_file.endswith('.xlsx'):
        # Load the Excel file
        xls = pd.ExcelFile(file_path)
        
        # Dropdown to select a sheet
        sheet_name = st.selectbox('Select a sheet', xls.sheet_names)
        
        if sheet_name:
            df = pd.read_excel(xls, sheet_name=sheet_name)
    else:
        # Read CSV file
        df = pd.read_csv(file_path)

    columns = df.columns.tolist()

    st.write("")

##################################################################################
# Check Type and primary statistics 

    st.header("🔍 Data Preview")

    head_df = df.head()
    dtypes_series = df.dtypes

    # Create a new row with the data types
    dtypes_row = pd.Series(dtypes_series.values, index=head_df.columns, name="Type")

    # Concatenate the head DataFrame and the data types row
    combined_df = pd.concat([pd.DataFrame(dtypes_row).T,head_df])

    # Apply styling to highlight the first row (data types) in yellow
    def highlight_first_row(s):
        return ["background-color: lightyellow" if s.name == "Type" else "" for _ in s]
    
    st.dataframe(combined_df.style.apply(highlight_first_row, axis=1))


    ##################################################################################


    import pandas as pd

    # Assuming df is your pandas DataFrame and columns is a list of column names
    st.header('🔧 Changing the Type')


    # Initialize lastState in session state
    if 'lastState' not in st.session_state:
        st.session_state.lastState = None  # Default to None initially

    # ⚠️ Display lastState only if changes were made
    if st.session_state.lastState is None:
        st.warning("⚠️ In order to make changes: select the column(s) and desired type then confirm ")  # Updated warning message


    # Adjust column width to align button correctly with dropdowns
    col1, col2, col3 = st.columns([3, 3, 1.5])  # Adjusted button column width

    # Initialize session state for df_dtypes if not already present
    if 'df_dtypes' not in st.session_state:
        st.session_state.df_dtypes = df.dtypes.to_dict()  # Initialize with current dtypes



    # Column selection
    with col1:
        choose_cols = st.multiselect('Change the type of specific columns', options=columns)

    # Type selection with default empty value if user hasn't made input yet
    with col2:
        possible_types = ['int64', 'float64', 'bool', 'datetime64[ns]', 'timedelta64[ns]', 'category']
        choose_type = st.selectbox('Set type to:', options=[""] + possible_types, index=0)  # Default empty

    # Fix button alignment and add controlled spacing
    with col3:
        st.markdown("<div style='margin-bottom: 6px; font-weight: bold;'>Confirm</div>", unsafe_allow_html=True)  # Adjusted spacing
        change_type_clicked = st.button("Change Type")  # Button right under the text



    # Ensure both column(s) and type are selected
    if choose_cols and choose_type and change_type_clicked:
        try:
            for col in choose_cols:
                if choose_type == 'category':
                    df[col] = df[col].astype('category')
                    
                else:
                    df[col] = df[col].astype(choose_type)

                # Update session state with the new dtype
                st.session_state.df_dtypes[col] = choose_type

            # ✅ Show success message when type change is confirmed
            st.success(f"✅ Column(s) '{', '.join(choose_cols)}' type successfully changed to '{choose_type}'.")

            # Create a new row with the updated data types
            dtypes_series = pd.Series(st.session_state.df_dtypes)
            dtypes_row = pd.Series(dtypes_series.values, index=head_df.columns, name="Type")

            # Concatenate the head DataFrame and the data types row
            combined_df = pd.concat([pd.DataFrame(dtypes_row).T, head_df])

            # Apply styling to highlight the first row (data types) in yellow
            def highlight_first_row(s):
                return ["background-color: lightyellow" if s.name == "Type" else "" for _ in s]
            
            st.dataframe(combined_df.style.apply(highlight_first_row, axis=1))


            # Store in session state to persist across reruns
            st.session_state.lastState = combined_df

        except ValueError as e:
            st.error(f"❌ Error changing type: {e}")
        except TypeError as e:
            st.error(f"❌ Error changing type: {e}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")


    ##################################################################################
    # --- START OF KPI SECTION (Modified Display Logic) ---

    st.header('🔢 Key Performance Indicators (KPIs)')

    # Initialize df_dtypes... (rest of the setup code remains here)
    if 'df_dtypes' not in st.session_state:
        try:
            st.session_state.df_dtypes = df.dtypes.to_dict()
        except NameError:
            st.error("Error: DataFrame 'df' is not loaded or defined yet. Cannot calculate KPIs.")
            st.stop() 

    # Allow the user to select the column
    select_options = ["--- Select a Field ---"] + columns
    kpi_column = st.selectbox('Select the field for KPI calculation', 
                            options=select_options, 
                            key='kpi_field_auto')

    if kpi_column == "--- Select a Field ---":
        st.info("Please select a data field above to view its key statistics.")

    elif kpi_column in columns:
        
        kpi_dtype = st.session_state.df_dtypes.get(kpi_column, df[kpi_column].dtype)

        if kpi_dtype == 'int64' or kpi_dtype == 'float64':

            s = df[kpi_column]

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

            # ROW 1 (3 columns): Valid, Missing, Sum
            col1, col2, col3 = st.columns(3)
            cols_row1 = [col1, col2, col3]

            for i in range(3):
                title, result = kpi_list[i]

                if isinstance(result, (float)):
                    display_value = f"{result:,.2f}"
                else:
                    display_value = str(result)

                with cols_row1[i]:
                    st.metric(label=f"{title} of {kpi_column}", value=display_value)

            # ROW 2 (3 columns): Min, Max, Mean
            col4, col5, col6 = st.columns(3)
            cols_row2 = [col4, col5, col6]

            for i in range(3):
                title, result = kpi_list[i + 3]

                if isinstance(result, (float, int)):
                    display_value = f"{result:,.2f}"
                else:
                    display_value = str(result)

                with cols_row2[i]:
                    st.metric(label=f"{title} of {kpi_column}", value=display_value)



        # --- DATE KPIs ---
        elif (
            "datetime64" in str(kpi_dtype)
            or str(kpi_dtype) == "datetime64[ns]"
            or str(kpi_dtype) == "datetime64[ns, UTC]"
            or kpi_dtype == "object"  # try-parse objects as dates
        ):
            # Try to coerce to datetime (safe for datetime cols too)
            s = pd.to_datetime(df[kpi_column], errors="coerce")

            valid = s.dropna()
            missing_count = s.isna().sum()

            if valid.empty:
                st.warning(
                    f"Column '{kpi_column}' looks like a date field, but no valid dates could be parsed."
                )
            else:
                min_dt = valid.min()
                max_dt = valid.max()
                span_days = (max_dt - min_dt).days
                unique_dates = valid.dt.date.nunique()

                # Optional: most frequent date (mode)
                mode_date = valid.dt.date.mode()
                mode_date_value = mode_date.iloc[0] if not mode_date.empty else None

                # KPI calculations (dates displayed nicely)
                kpi_calculations = {
                    "Count (valid)": int(valid.shape[0]),
                    "Missing": int(missing_count),
                    "Unique": int(unique_dates),
                    "Min": min_dt.date(),
                    "Max": max_dt.date(),
                    "Span (days)": int(span_days),
                }


                # If you want a 7th metric (to fill 2 rows of 3 + 3 nicely, you already have 6)
                # You can swap one in or display it below:
                # kpi_calculations["Most common date"] = mode_date_value

                kpi_list = list(kpi_calculations.items())

                # ROW 1 (3 columns)
                col1, col2, col3 = st.columns(3)
                cols_row1 = [col1, col2, col3]

                for i in range(3):
                    title, result = kpi_list[i]
                    display_value = str(result)
                    with cols_row1[i]:
                        st.metric(label=f"{title} of {kpi_column}", value=display_value)

                # ROW 2 (3 columns)
                col4, col5, col6 = st.columns(3)
                cols_row2 = [col4, col5, col6]

                for i in range(3):
                    title, result = kpi_list[i + 3]
                    display_value = str(result)
                    with cols_row2[i]:
                        st.metric(label=f"{title} of {kpi_column}", value=display_value)

                # Optional extra info (mode)
                if mode_date_value is not None:
                    st.caption(f"Most common date: {mode_date_value}")

        # --- CATEGORY KPIs ---
        elif str(kpi_dtype) == "category":

            s = df[kpi_column]

            valid_count = int(s.notna().sum())
            missing_count = int(s.isna().sum())
            unique_count = int(s.nunique(dropna=True))

            kpi_calculations = {
                "Count (valid)": valid_count,
                "Missing": missing_count,
                "Unique": unique_count,
            }

            kpi_list = list(kpi_calculations.items())

            # Single row (3 columns)
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]

            for i in range(3):
                title, result = kpi_list[i]

                # Counts should never be formatted
                display_value = str(result)

                with cols[i]:
                    st.metric(label=f"{title} of {kpi_column}", value=display_value)



        else:
            st.warning(f"The column '{kpi_column}' is not numeric ({kpi_dtype}). Select a numeric field for KPIs.")
            
    # --- END OF KPI SECTION ---


    ##################################################################################
    # Create Visualisation 
    st.header('📊 Charts & Visuals')

    # Assuming df is your pandas DataFrame and columns is a list of column names

    # Initialize with current dtypes
    if 'df_dtypes' not in st.session_state:
        st.session_state.df_dtypes = df.dtypes.to_dict()  

    # Allow the user to select columns for plotting
    x_axis = st.selectbox('Select the X-axis', options=columns + ["None"])
    y_axis = st.selectbox('Select the Y-axis', options=columns + ["None"])


    aggregation_functions = ["sum","average","count"]
    z_axis = "None"
    plot_list = []  

    if x_axis != "None" and y_axis != "None":
        # Get dtypes from session state, or default to df.dtypes if not available
        x_dtype = st.session_state.df_dtypes.get(x_axis, df[x_axis].dtype)
        y_dtype = st.session_state.df_dtypes.get(y_axis, df[y_axis].dtype)

        #possible plots depending on data type
        if x_dtype == 'category' and (y_dtype == 'int64' or y_dtype == 'float64'):
            plot_list.append('Bar Chart')  # Add Bar Chart if conditions are metplot_list.append('Bar Chart') 
            plot_list.append('Pie Chart') 
            plot_list.append('Box Chart') 

        elif (x_dtype == 'int64' or x_dtype == 'float64') and (y_dtype == 'int64' or y_dtype == 'float64'):
            plot_list.append('Scatter Plot')  
            plot_list.append('Line Chart')  

        elif (x_dtype == 'datetime64[ns]' ):
            plot_list.append('Line Chart')  

        elif ( x_dtype == 'category' ) and (y_dtype == 'object' ):
            # Allow the user to select columns for plotting
            z_axis = st.selectbox('Select the aggregation funtion', options=aggregation_functions + ["None"])
            plot_list.append('Bar Chart')  


    # If none have been selected, reset the plot list.
    if x_axis == "None" or y_axis == "None":
        plot_list = ['Line Chart', 'Scatter Plot', 'Distribution Plot', 'Count Plot']

    plot_type = st.selectbox('Select the type of plot', options=plot_list)


    # Generate the plot based on user selection
    if st.button('Generate Plot'):

        fig, ax = plt.subplots(figsize=(6, 4))


        if plot_type == 'Bar Chart' and z_axis == "count":

            #plotting.bar_count(df,x_axis, plot_type, aggregation_functions)

            # Group by x_axis and count occurrences in y_axis
            count_df = df.groupby(x_axis).size().reset_index(name='count')
            count_df = count_df.sort_values(by='count', ascending=False)

            # Plot the barplot with count
            sns.barplot(x=count_df[x_axis], y=count_df['count'], ax=ax)

            # Rotate x-axis labels vertically
            ax.tick_params(axis='x', rotation=90)  # Rotate labels 90 degrees

            # Optionally, you can also display the count on top of the bars for better clarity
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')


        elif plot_type == 'Bar Chart' and z_axis != "count":
            sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif plot_type == 'Box Chart':
            a=2
            ##sns.boxplot(x=df[x_axis], y=df[y_axis], ax=ax)  
        elif plot_type == "Pie Chart":
            pie_data = df.groupby(x_axis)[y_axis].sum()  # Aggregate values
            ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("pastel"))

        elif plot_type == 'Line Chart':
            if(x_dtype == 'datetime64[ns]'):
                plotting.create_lineChart_Date(df,x_axis, y_axis, ax)
            else:
                sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif plot_type == 'Scatter Plot':
            sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)


        elif plot_type == 'Distribution Plot':
            sns.histplot(df[x_axis], kde=True, ax=ax)
            y_axis='Density'
        elif plot_type == 'Count Plot':
            sns.countplot(x=df[x_axis], ax=ax)
            y_axis = 'Count'


                   
        # Adjust label sizes
        ax.tick_params(axis='x', labelsize=10)  # Adjust x-axis label size
        ax.tick_params(axis='y', labelsize=10)  # Adjust y-axis label size

        # Adjust title and axis labels with a smaller font size
        plt.title(f'{plot_type} of {y_axis} vs {x_axis}', fontsize=12)
        plt.xlabel(x_axis, fontsize=10)
        plt.ylabel(y_axis, fontsize=10)

        # Show the results
        st.pyplot(fig)
