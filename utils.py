# utils.py
"""
Utility module: Logging, caching, exceptions, progress, and data helpers.
"""

import structlog
import warnings
import numpy as np
import pandas as pd
from joblib import Memory
from typing import List
from tensorflow.keras.callbacks import EarlyStopping
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn  # For non-Streamlit; adapt in app
from sklearn.feature_selection import mutual_info_regression  # Added for MI

# Suppress TensorFlow warnings for SHAP
warnings.filterwarnings("ignore", category=UserWarning, module="shap.explainers._deep.deep_tf")

# Setup logging
structlog.configure(
    processors=[structlog.processors.JSONRenderer()],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger()

# Cache for computations (joblib; complement with st.cache_data in app)
memory = Memory("cache_dir", verbose=0)

class MissingDataError(Exception):
    """Raised when insufficient data is provided."""
    pass

class InvalidPressureError(Exception):
    """Raised when pressure column is invalid."""
    pass

class RichProgressCallback(EarlyStopping):
    """Custom Keras callback for training progress with rich (adapt for st.progress in app)."""
    def __init__(self, total_epochs: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.total_epochs: int = total_epochs
        self.progress: Progress | None = None
        self.task_id = None

    def on_train_begin(self, logs=None) -> None:
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("Loss: {task.fields[loss]:.4f} | Val Loss: {task.fields[val_loss]:.4f}")
        )
        self.progress.__enter__()
        self.task_id = self.progress.add_task("Training Neural Network", total=self.total_epochs, loss=0.0, val_loss=0.0)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if self.progress and self.task_id:
            self.progress.update(
                self.task_id,
                advance=1,
                loss=logs.get('loss', 0),
                val_loss=logs.get('val_loss', 'N/A')
            )

    def on_train_end(self, logs=None) -> None:
        if self.progress:
            self.progress.__exit__(None, None, None)

def handle_outliers(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Handle outliers using IQR method."""
    for param in columns:
        if df[param].dtype not in [np.float64, np.int64]:
            logger.warning(f"Skipping outlier handling for non-numeric column {param}")
            continue
        Q1 = df[param].quantile(0.25)
        Q3 = df[param].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[param] = df[param].clip(lower=lower_bound, upper=upper_bound)
        logger.info("Clipped outliers", param=param, lower=lower_bound, upper=upper_bound)
    return df

def impute_missing_values(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Impute missing values with median and drop columns with all NaNs or non-numeric data."""
    valid_columns = []
    for param in columns:
        if df[param].isna().all():
            logger.warning(f"Dropping column {param}: all values are NaN")
            continue
        if df[param].dtype not in [np.float64, np.int64]:
            logger.warning(f"Dropping column {param}: non-numeric data")
            continue
        missing_ratio = df[param].isna().sum() / len(df)
        if missing_ratio > 0.5:
            logger.warning(f"Column {param} has {missing_ratio:.2%} missing values")
        median_value = df[param].median()
        if np.isnan(median_value):
            logger.warning(f"Dropping column {param}: median is NaN")
            continue
        df[param] = df[param].fillna(median_value)
        if df[param].isna().any():
            logger.warning(f"Dropping column {param}: still contains NaNs after imputation")
            continue
        valid_columns.append(param)
    df = df[valid_columns]
    logger.info("Imputed missing values", valid_columns=valid_columns)
    return df

def compute_mi_score(param1: str, param2: str, df: pd.DataFrame, random_seed: int = 42) -> float:
    """Compute mutual information score for a parameter pair."""
    if param1 == param2:
        return 1.0
    X = df[[param1]].fillna(df[param1].median()).values  # Shape: (n_samples, 1)
    y = df[param2].fillna(df[param2].median()).values.ravel()  # Shape: (n_samples,)
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        logger.error(f"NaN values detected in {param1} or {param2}")
        raise ValueError(f"Input contains NaN for {param1} or {param2}")
    rng = np.random.default_rng(random_seed)
    score = mutual_info_regression(X, y, random_state=rng.integers(0, 2**32))[0]
    logger.debug("Computed MI score", param1=param1, param2=param2, score=score)
    return score

def compute_mi_matrix(df: pd.DataFrame, params: List[str], regime: str | None = None, config=None) -> pd.DataFrame:
    """Compute mutual information matrix sequentially."""
    if config is None:
        class DummyConfig:
            min_rows = 10
            random_seed = 42
        config = DummyConfig()
    if len(df) < config.min_rows:
        logger.warning(f"Dataset has {len(df)} rows, minimum is {config.min_rows}, returning empty matrix")
        return pd.DataFrame(np.nan, index=params, columns=params)
    
    mi_matrix = np.zeros((len(params), len(params)))
    param_pairs = [(i, j, params[i], params[j]) for i in range(len(params)) for j in range(i, len(params))]
    failed_pairs = []
    
    # Simplified progress (no rich; use st.progress in app)
    for i, j, param1, param2 in param_pairs:
        try:
            score = compute_mi_score(param1, param2, df, config.random_seed)
            mi_matrix[i, j] = score
            mi_matrix[j, i] = score
        except ValueError as e:
            logger.error("MI computation failed", param1=param1, param2=param2, error=str(e))
            failed_pairs.append((param1, param2))
            mi_matrix[i, j] = 0.0
            mi_matrix[j, i] = 0.0
    
    mi_matrix = pd.DataFrame(mi_matrix, index=params, columns=params)
    max_score = mi_matrix.max().max()
    if max_score > 0:
        mi_matrix = mi_matrix / max_score
    else:
        logger.warning(f"MI matrix ({regime or 'all'}) has max score 0, skipping normalization")
    logger.info("Computed MI matrix", regime=regime or "all", shape=mi_matrix.shape)
    return mi_matrix
