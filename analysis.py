# analysis.py
"""
Analysis module: MI, correlations, RF importance, SHAP.
Orchestrates relationship analysis; returns matrices/dicts for viz.
Preserves regime splits if pressure detected.
Supports multiple ML methods (rf, nn, lr, svm) for importance.
"""

import streamlit as st  # For progress bars
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from typing import List, Dict, Tuple, Optional
from config import Config
from utils import compute_mi_score, compute_mi_matrix, logger, MissingDataError, memory

@memory.cache
def compute_mi_matrix_cached(df: pd.DataFrame, params: List[str], regime: Optional[str] = None, config: Config | None = None) -> pd.DataFrame:
    """Cached wrapper for MI matrix."""
    return compute_mi_matrix(df, params, regime, config)

def analyze_relationships(
    df: pd.DataFrame, 
    params: List[str], 
    pressure_col: Optional[str], 
    bubble_point: float, 
    config: Config, 
    ml_method: str = "rf",  # Added: ML method param
    mi_params: Optional[List[str]] = None
) -> Dict[str, any]:
    """Perform relationship analysis; returns dict with matrices, importance, SHAP for viz."""
    df_classified = df.copy()
    if pressure_col:
        from data import classify_regimes
        df_classified = classify_regimes(df, pressure_col, bubble_point)

    mi_params = mi_params or params
    if not all(p in params for p in mi_params):
        raise ValueError("mi_params must be a subset of params")
    if len(mi_params) > 10:
        logger.warning("Large number of MI parameters may be slow", count=len(mi_params))

    # Progress for MI computation (adapted from original loop)
    progress_mi = st.progress(0)
    status_mi = st.empty()
    status_mi.text("Computing correlations and MI...")
    
    corr_matrix = df_classified[params].corr(method='spearman')  # Original Spearman
    progress_mi.progress(0.2)
    logger.info("Computed correlation matrix", shape=corr_matrix.shape)

    mi_matrix = pd.DataFrame(np.nan, index=mi_params, columns=mi_params)
    try:
        mi_matrix = compute_mi_matrix_cached(df_classified, mi_params, config=config)
        progress_mi.progress(0.4)
    except Exception as e:
        logger.warning("Failed to compute MI matrix", error=str(e))

    mi_saturated = pd.DataFrame(np.nan, index=mi_params, columns=mi_params)
    mi_undersaturated = pd.DataFrame(np.nan, index=mi_params, columns=mi_params)
    if pressure_col:
        try:
            saturated_df = df_classified[df_classified['Regime'] == 'Saturated']
            if len(saturated_df) > 0:
                mi_saturated = compute_mi_matrix_cached(saturated_df, mi_params, regime='saturated', config=config)
                progress_mi.progress(0.6)
        except Exception as e:
            logger.warning("Failed to compute MI saturated matrix", error=str(e))
        try:
            undersaturated_df = df_classified[df_classified['Regime'] == 'Undersaturated']
            if len(undersaturated_df) > 0:
                mi_undersaturated = compute_mi_matrix_cached(undersaturated_df, mi_params, regime='undersaturated', config=config)
                progress_mi.progress(0.8)
        except Exception as e:
            logger.warning("Failed to compute MI undersaturated matrix", error=str(e))
    
    progress_mi.progress(1.0)
    progress_mi.empty()
    status_mi.empty()

    # Model selection for importance (original RF, extended to others)
    model_class = {
        "rf": RandomForestRegressor,
        "nn": None,  # NN not for importance here; use RF as fallback or skip
        "lr": LinearRegression,
        "svm": SVR,
        "dt": DecisionTreeRegressor,
        "ridge": lambda: LinearRegression(fit_intercept=True, positive=False)  # Ridge via alpha param if needed
    }.get(ml_method, RandomForestRegressor)
    
    # Random Forest/Alternative importance
    rf_importance: Dict[str, pd.DataFrame] = {}
    shap_values_dict: Dict[str, Tuple[np.ndarray, pd.DataFrame]] = {}
    rng = np.random.default_rng(config.random_seed)
    num_targets = len(params)
    progress_importance = st.progress(0)
    status_importance = st.empty()
    status_importance.text(f"Computing {ml_method.upper()} importance...")
    
    for i, target in enumerate(params):
        X = df_classified[params].drop(columns=[target])
        y = df_classified[target]
        if y.isna().any() or len(X) < config.min_rows:
            logger.warning(f"Skipping {ml_method} for {target}: NaNs or insufficient rows")
            continue
        if ml_method == "nn":
            # For NN, train per target (adapted from original opt NN)
            model, _ = train_neural_network(X, y, config)  # Use trained NN
            # Importance from weights (original formula)
            weights = model.layers[0].get_weights()[0]
            perm_importance = np.mean(np.abs(weights), axis=1)  # Original mean abs weights
        else:
            model = model_class(n_jobs=config.n_jobs if hasattr(model_class, 'n_jobs') else None)
            if hasattr(model, 'random_state'):
                model.random_state = rng.integers(0, 2**32)
            model.fit(X, y)
            perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=rng.integers(0, 2**32), n_jobs=config.n_jobs)
            perm_importance = perm_importance.importances_mean
        
        rf_importance[target] = pd.DataFrame({
            'Feature': X.columns,
            'Importance': perm_importance
        }).sort_values('Importance', ascending=False)
        
        progress_importance.progress((i + 1) / num_targets)
        
        if 'shap' in config.visualizations:
            try:
                from models import compute_shap_values
                shap_values, X_sample = compute_shap_values(model, X, config.shap_sample_size, config.random_seed)
                shap_values_dict[target] = (shap_values, X_sample)
            except Exception as e:
                logger.error(f"Failed SHAP for {target}", error=str(e))
    
    progress_importance.empty()
    status_importance.empty()

    results = {
        'corr_matrix': corr_matrix,
        'mi_matrix': mi_matrix,
        'mi_saturated': mi_saturated,
        'mi_undersaturated': mi_undersaturated,
        'rf_importance': rf_importance,  # Rename to importance if needed
        'shap_values_dict': shap_values_dict,
        'df': df_classified  # For viz
    }
    logger.info("Analysis complete", keys=list(results.keys()))
    return results
