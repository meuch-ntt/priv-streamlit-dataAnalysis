# dataset_selector.py
from __future__ import annotations

import os
from typing import Tuple

import streamlit as st

import data_loader


def ensure_dataset_session_state() -> None:
    """Initialize lightweight dataset selection + per-dataset override storage."""
    if "loaded_file" not in st.session_state:
        st.session_state.loaded_file = None
    if "loaded_sheet" not in st.session_state:
        st.session_state.loaded_sheet = None

    if "overrides_by_dataset" not in st.session_state:
        st.session_state.overrides_by_dataset = {}


def get_dataset_state(dataset_key: tuple[str, str | None]):
    """Per-dataset storage for type overrides and semantic overrides."""
    entry = st.session_state.overrides_by_dataset.get(dataset_key)
    if entry is None:
        entry = {"type_overrides": {}, "column_semantics": {}}
        st.session_state.overrides_by_dataset[dataset_key] = entry
    return entry["type_overrides"], entry["column_semantics"]


def select_dataset_ui(
    folder_path: str,
    data_file_exts: tuple[str, ...],
    ext_csv: str,
    ext_xlsx: str,
    msg_folder_not_found_prefix: str,
    msg_no_files_found: str,
):
    """
    UI + orchestration:
    - validate folder
    - list/select file
    - if xlsx: list/select sheet
    - manage lightweight session state for currently loaded dataset
    - return dataset identity + token
    """
    ensure_dataset_session_state()

    if not os.path.isdir(folder_path):
        st.error(f"{msg_folder_not_found_prefix} {folder_path}")
        st.stop()

    files = data_loader.list_data_files(folder_path, data_file_exts)
    if not files:
        st.warning(msg_no_files_found)
        st.stop()

    selected_file = st.selectbox("Select a file", files, index=None)
    if not selected_file:
        st.info("First, select the file with the data you want to analyze.")
        st.stop()

    file_path = os.path.join(folder_path, selected_file)
    token = data_loader.file_token(file_path)
    ext = ext_xlsx if selected_file.endswith(ext_xlsx) else ext_csv

    sheet_name: str | None = None

    if ext == ext_xlsx:
        try:
            sheet_names = data_loader.list_excel_sheets_cached(file_path, token)
            if not sheet_names:
                st.error("No sheets found in this Excel file.")
                st.stop()

            sheet_name = st.selectbox("Select a sheet", sheet_names, index=None)
            if sheet_name is None:
                st.stop()

            is_new_dataset = (
                st.session_state.loaded_file != selected_file
                or st.session_state.loaded_sheet != sheet_name
            )
            if is_new_dataset:
                st.session_state.loaded_file = selected_file
                st.session_state.loaded_sheet = sheet_name

        except Exception as e:
            st.error("❌ Failed to read the Excel file. Please check the file and selected sheet.")
            st.exception(e)
            st.stop()

    else:
        is_new_dataset = (
            st.session_state.loaded_file != selected_file
            or st.session_state.loaded_sheet is not None
        )
        if is_new_dataset:
            st.session_state.loaded_file = selected_file
            st.session_state.loaded_sheet = None

    dataset_key = (selected_file, sheet_name)
    return selected_file, file_path, ext, sheet_name, token, dataset_key
