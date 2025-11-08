##################################### 
# KPI Session as dropdown

"""
st.header('📊 Key Performance Indicators (KPIs)')

# Initialize df_dtypes in session state only if it doesn't exist.
if 'df_dtypes' not in st.session_state:
    try:
        # Check if df is defined before accessing its attributes
        st.session_state.df_dtypes = df.dtypes.to_dict()
    except NameError:
        st.error("Error: DataFrame 'df' is not loaded or defined yet. Cannot calculate KPIs.")
        st.stop() 

# Allow the user to select the column for KPI calculation
# The key is changed to ensure the widget resets correctly if needed
kpi_column = st.selectbox('Select the field for KPI calculation', options=columns, key='kpi_field_auto')

# Get the data type for the selected column
kpi_dtype = st.session_state.df_dtypes.get(kpi_column, df[kpi_column].dtype)

# --- Calculation and Display ---

# Only proceed if the selected column is numeric (int64 or float64)
if kpi_dtype == 'int64' or kpi_dtype == 'float64':
    
    # 1. Define the KPI functions to calculate
    # We will compute ALL of these at once
    kpi_functions = {
        'Sum': df[kpi_column].sum(),
        'Mean': df[kpi_column].mean(),
        'Median': df[kpi_column].median(),
        'Min': df[kpi_column].min(),
        'Max': df[kpi_column].max()
    }

    # 2. Define the columns layout (e.g., 5 metrics side-by-side)
    cols = st.columns(len(kpi_functions))

    # 3. Loop through the calculations and display in columns
    for i, (title, result) in enumerate(kpi_functions.items()):
        
        # Ensure the result is a number before formatting
        if isinstance(result, (float, int)):
            display_value = f"{result:,.2f}" 
        else:
            display_value = str(result)
            
        with cols[i]:
            st.metric(label=f"{title} of {kpi_column}", value=display_value)

else:
    # If a non-numeric column is selected, show a message
    st.info(f"The column '{kpi_column}' is not numeric ({kpi_dtype}). Only numeric columns can calculate Sum, Mean, Min, etc.")
    
# --- END OF KPI SECTION ---
"""