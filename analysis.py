# analysis.py
"""
Analysis module: MI, correlations, RF importance, SHAP.
Orchestrates relationship analysis; returns matrices/dicts for viz.
Preserves regime splits if pressure detected.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from typing import List, Dict, Tuple, Optional
from config import Config
from utils import compute_mi_score, compute_mi_matrix, logger, MissingDataError, memory  # Added memory

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
    mi_params: Optional[List[str]] = None
) -> Dict[str, any]:
    """Perform relationship analysis; returns dict with matrices, importance, SHAP for viz."""
    df_classified = df.copy()
    if pressure_col:
        from data import classify_regimes  # Local import
        df_classified = classify_regimes(df, pressure_col, bubble_point)

    mi_params = mi_params or params
    if not all(p in params for p in mi_params):
        raise ValueError("mi_params must be a subset of params")
    if len(mi_params) > 10:
        logger.warning("Large number of MI parameters may be slow", count=len(mi_params))

    # Compute correlations and MI
    corr_matrix = df_classified[params].corr(method='spearman')
    logger.info("Computed correlation matrix", shape=corr_matrix.shape)

    mi_matrix = pd.DataFrame(np.nan, index=mi_params, columns=mi_params)
    try:
        mi_matrix = compute_mi_matrix_cached(df_classified, mi_params, config=config)
    except Exception as e:
        logger.warning("Failed to compute MI matrix", error=str(e))

    mi_saturated = pd.DataFrame(np.nan, index=mi_params, columns=mi_params)
    mi_undersaturated = pd.DataFrame(np.nan, index=mi_params, columns=mi_params)
    if pressure_col:
        try:
            saturated_df = df_classified[df_classified['Regime'] == 'Saturated']
            if len(saturated_df) > 0:
                mi_saturated = compute_mi_matrix_cached(saturated_df, mi_params, regime='saturated', config=config)
        except Exception as e:
            logger.warning("Failed to compute MI saturated matrix", error=str(e))
        try:
            undersaturated_df = df_classified[df_classified['Regime'] == 'Undersaturated']
            if len(undersaturated_df) > 0:
                mi_undersaturated = compute_mi_matrix_cached(undersaturated_df, mi_params, regime='undersaturated', config=config)
        except Exception as e:
            logger.warning("Failed to compute MI undersaturated matrix", error=str(e))

    # Random Forest importance
    rf_importance: Dict[str, pd.DataFrame] = {}
    shap_values_dict: Dict[str, Tuple[np.ndarray, pd.DataFrame]] = {}
    rng = np.random.default_rng(config.random_seed)  # Fixed: Direct RNG, no rng_seed import
    for target in params:
        X = df_classified[params].drop(columns=[target])
        y = df_classified[target]
        if y.isna().any() or len(X) < config.min_rows:
            logger.warning(f"Skipping RF for {target}: NaNs or insufficient rows")
            continue
        rf = RandomForestRegressor(n_estimators=100, random_state=rng.integers(0, 2**32), n_jobs=config.n_jobs)
        rf.fit(X, y)
        perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=rng.integers(0, 2**32), n_jobs=config.n_jobs)
        rf_importance[target] = pd.DataFrame({
            'Feature': X.columns,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)
        if 'shap' in config.visualizations:
            try:
                from models import compute_shap_values
                shap_values, X_sample = compute_shap_values(rf, X, config.shap_sample_size, config.random_seed)
                shap_values_dict[target] = (shap_values, X_sample)
            except Exception as e:
                logger.error(f"Failed SHAP for {target}", error=str(e))

    results = {
        'corr_matrix': corr_matrix,
        'mi_matrix': mi_matrix,
        'mi_saturated': mi_saturated,
        'mi_undersaturated': mi_undersaturated,
        'rf_importance': rf_importance,
        'shap_values_dict': shap_values_dict,
        'df': df_classified  # Added: For viz access
    }
    logger.info("Analysis complete", keys=list(results.keys()))
    return results
