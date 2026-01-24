'''deleted

# ==============================================================================
# Changing the Type
# ==============================================================================

st.subheader("🔧 Changing the Type")

# show info until user clicked "Change Type" at least once
if "change_type_has_clicked" not in st.session_state:
    st.session_state.change_type_has_clicked = False

info_slot = st.empty()
if not st.session_state.change_type_has_clicked:
    info_slot.info("Optionally force a different type if it was not interpreted correctly.")

# reset widgets after successful change
if st.session_state.get("_reset_change_type_widgets", False):
    st.session_state["change_type_cols"] = []
    st.session_state["change_type_target"] = None
    st.session_state["_reset_change_type_widgets"] = False

col1, col2, col3 = st.columns([3, 3, 1.5])

with col1:
    choose_cols = st.multiselect(
        "Change the type of specific columns",
        options=columns,
        key="change_type_cols",
    )

with col2:
    target_type = st.selectbox(
        "Set type to:",
        options=TARGET_OPTIONS,
        index=None,
        placeholder="Choose an option",
        key="change_type_target",
    )

with col3:
    st.markdown(
        "<div style='margin-bottom: 6px; font-weight: bold;'>Confirm</div>",
        unsafe_allow_html=True,
    )
    change_type_clicked = st.button("Change Type")

if choose_cols and target_type and change_type_clicked:
    st.session_state.change_type_has_clicked = True
    info_slot.empty()

    try:
        for col in choose_cols:
            if target_type == TARGET_DATE:
                type_overrides[col] = TARGET_DATETIME
                semantics_overrides[col] = SEM_DATE
            else:
                type_overrides[col] = target_type
                semantics_overrides.pop(col, None)

        # Persist overrides per dataset
        if hasattr(dataset_selector, "set_dataset_state"):
            dataset_selector.set_dataset_state(dataset_key, type_overrides, semantics_overrides)

        st.session_state["change_type_success_msg"] = (
            f'✅ Column(s) "{", ".join(choose_cols)}" type changed successfully. '
            "See data preview above."
        )
        st.session_state["change_type_success"] = True
        st.session_state["_reset_change_type_widgets"] = True
        st.rerun()

    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")

if st.session_state.get("change_type_success"):
    st.success(st.session_state["change_type_success_msg"])
    st.session_state["change_type_success"] = False


'''





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