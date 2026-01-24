# type_editor.py
from __future__ import annotations

import streamlit as st


# ==============================================================================
# Type editor UI
# ==============================================================================

def set(
    *,
    columns: list[str],
    target_options: list[str],
    target_datetime_value: str,
    target_date_value: str,
    sem_date_value: str,
    type_overrides: dict[str, str],
    column_semantics: dict[str, str],
    cols_key: str = "change_type_cols",
    target_key: str = "change_type_target",
    reset_flag_key: str = "_reset_change_type_widgets",
) -> None:
    """
    Streamlit UI to update type overrides and semantic overrides for the current dataset.
    """
    st.subheader("🔧 Changing the Type")

    if "change_type_has_clicked" not in st.session_state:
        st.session_state.change_type_has_clicked = False

    info_slot = st.empty()
    if not st.session_state.change_type_has_clicked:
        info_slot.info("Change the data type of columns if they were not interpreted correctly.")

    if st.session_state.get(reset_flag_key, False):
        st.session_state[cols_key] = []
        st.session_state[target_key] = None
        st.session_state[reset_flag_key] = False

    col1, col2, col3 = st.columns([3, 3, 1.5])

    with col1:
        choose_cols = st.multiselect(
            "Change the type of specific columns",
            options=columns,
            key=cols_key,
        )

    with col2:
        target_type = st.selectbox(
            "Set type to:",
            options=target_options,
            index=None,
            placeholder="Choose an option",
            key=target_key,
        )

    with col3:
        st.markdown("<div style='margin-bottom: 6px; font-weight: bold;'>Confirm</div>", unsafe_allow_html=True)
        change_type_clicked = st.button("Change Type")

    if choose_cols and target_type and change_type_clicked:
        st.session_state.change_type_has_clicked = True
        info_slot.empty()

        try:
            for col in choose_cols:
                if target_type == target_date_value:
                    type_overrides[col] = target_datetime_value
                    column_semantics[col] = sem_date_value
                else:
                    type_overrides[col] = target_type
                    column_semantics.pop(col, None)

            st.session_state["change_type_success_msg"] = (
                f'✅ Column(s) "{", ".join(choose_cols)}" type changed successfully. See data preview above.'
            )
            st.session_state["change_type_success"] = True
            st.session_state[reset_flag_key] = True
            st.rerun()

        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")

    if st.session_state.get("change_type_success"):
        st.success(st.session_state["change_type_success_msg"])
        st.session_state["change_type_success"] = False
