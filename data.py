# data.py
"""
Data loading and preprocessing module.
Handles Excel uploads (via Streamlit), sample data, outlier/impute, and regime classification.
Pressure detection: Looks for 'P' or 'pressure' (case-insensitive).
"""

import pandas as pd
import numpy as np
import io
import streamlit as st
from typing import Tuple, List, Optional
from config import Config
from utils import handle_outliers, impute_missing_values, logger, MissingDataError, InvalidPressureError

# No hardcoded potential_params; use all numerics by default, user selects in app

def create_sample_data(rng: np.random.Generator, n_samples: int = 100) -> pd.DataFrame:
    """Create generic sample data (adaptable; no domain-specific columns)."""
    # Generic columns for demo; in app, base on user-selected names
    return pd.DataFrame({
        'Feature1': 1.2 + rng.normal(0, 0.05, n_samples),
        'Feature2': 500 + rng.normal(0, 50, n_samples),
        'P': 1900 + rng.uniform(-400, 400, n_samples),  # Include pressure for demo
        'Feature3': 30 + rng.normal(0, 2, n_samples),
        'Feature4': 0.7 + rng.normal(0, 0.02, n_samples),
        'Feature5': 150 + rng.normal(0, 5, n_samples)
    })

def load_and_preprocess_data(
    uploaded_files: List[st.runtime.uploaded_file.UploadedFile], 
    n_rows: Optional[int], 
    config: Config, 
    rng: np.random.Generator
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and preprocess data from Streamlit uploaded Excel files or use sample data."""
    if not uploaded_files:
        st.info("No files uploaded. Using sample data.")
        return create_sample_data(rng), list(create_sample_data(rng).columns)

    dfs = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        try:
            df_temp = pd.read_excel(io.BytesIO(uploaded_file.read()), sheet_name=0, engine='openpyxl')
            numeric_cols = df_temp.select_dtypes(include=[np.float64, np.int64]).columns.tolist()
            if not numeric_cols:
                logger.warning("No numeric columns found", file=uploaded_file.name)
                continue
            df_temp = df_temp[numeric_cols]
            df_temp = handle_outliers(df_temp, numeric_cols)
            df_temp = impute_missing_values(df_temp, numeric_cols)
            if df_temp.empty:
                logger.warning("No valid columns after preprocessing", file=uploaded_file.name)
                continue
            if n_rows is not None and n_rows < len(df_temp):
                df_temp = df_temp.sample(n=n_rows, random_state=rng.integers(0, 2**32))
                logger.info("Sampled rows", file=uploaded_file.name, n_rows=n_rows)
            dfs.append(df_temp)
        except Exception as e:
            logger.error("Error processing file", file=uploaded_file.name, error=str(e))
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        progress_bar.progress((i + 1) / len(uploaded_files))

    if not dfs:
        st.warning("No valid files loaded. Using sample data.")
        return create_sample_data(rng), []

    merged_df = pd.concat(dfs, ignore_index=True)
    if len(merged_df) < config.min_rows:
        raise MissingDataError(f"Merged dataset has {len(merged_df)} rows, but minimum is {config.min_rows}")

    # Remove domain-specific calc; no 'oil SG' handling

    params = list(merged_df.select_dtypes(include=[np.float64, np.int64]).columns)
    if not params:
        raise MissingDataError("No valid numeric parameters found after preprocessing")
    merged_df = impute_missing_values(merged_df, params)
    logger.info("Detected parameters", params=params)
    return merged_df, params

def detect_pressure_column(params: List[str]) -> Optional[str]:
    """Detect pressure column: 'P' or 'pressure' (case-insensitive)."""
    for col in params:
        if col.lower() in ['p', 'pressure']:
            return col
    return None

def classify_regimes(
    df: pd.DataFrame, 
    pressure_col: str, 
    bubble_point: float
) -> pd.DataFrame:
    """Classify data into saturated/undersaturated regimes based on pressure."""
    if pressure_col not in df.columns:
        raise InvalidPressureError(f"Pressure column '{pressure_col}' not found")
    df['Regime'] = np.where(df[pressure_col] >= bubble_point, 'Saturated', 'Undersaturated')
    logger.info("Classified regimes", pressure_col=pressure_col, bubble_point=bubble_point, 
                saturated=len(df[df['Regime'] == 'Saturated']), 
                undersaturated=len(df[df['Regime'] == 'Undersaturated']), 
                p_min=df[pressure_col].min(), p_max=df[pressure_col].max(), 
                p_mean=df[pressure_col].mean(), p_std=df[pressure_col].std())
    return df
