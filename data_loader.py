# data_loader.py
import os
import pandas as pd
import streamlit as st


# ==============================================================================
# Caching helpers (file listing + file reads)
# ==============================================================================

@st.cache_data(show_spinner=False)
def list_data_files(folder_path: str, exts: tuple[str, ...]) -> list[str]:
    """Return sorted list of files in a folder matching allowed extensions."""
    return sorted([f for f in os.listdir(folder_path) if f.endswith(exts)])


@st.cache_data(show_spinner=False)
def file_token(path: str) -> tuple[float, int]:
    """
    Cache-buster token: changes when file content changes.
    Using (mtime, size) is a good practical fingerprint for local apps.
    """
    return (os.path.getmtime(path), os.path.getsize(path))


@st.cache_data(show_spinner=True)
def read_csv_cached(path: str, token: tuple[float, int]) -> pd.DataFrame:
    """Cached CSV reader. 'token' exists only to invalidate cache when file changes."""
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def list_excel_sheets_cached(path: str, token: tuple[float, int]) -> list[str]:
    """Cached sheet name listing for Excel files."""
    with pd.ExcelFile(path) as xls:
        return xls.sheet_names


@st.cache_data(show_spinner=True)
def read_excel_sheet_cached(path: str, sheet_name: str, token: tuple[float, int]) -> pd.DataFrame:
    """Cached Excel reader for a specific sheet."""
    return pd.read_excel(path, sheet_name=sheet_name)


@st.cache_data(show_spinner=True)
def load_raw_df_cached(
    path: str,
    ext: str,
    sheet_name: str | None,
    token: tuple[float, int],
) -> pd.DataFrame:
    """
    Cached raw load entry point. This keeps raw loading logic in one place.
    """
    if ext == ".xlsx":
        if sheet_name is None:
            raise ValueError("sheet_name is required for Excel files.")
        return read_excel_sheet_cached(path, sheet_name, token)

    return read_csv_cached(path, token)
