# models.py
"""
ML Models module: NN building, training, predictions, SHAP.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from typing import Tuple
from config import Config
from utils import RichProgressCallback, memory, logger, MissingDataError

def build_neural_network(input_dim: int, config: Config) -> Sequential:
    """Build a dynamic neural network."""
    model = Sequential()
    for i, units in enumerate(config.nn_layers):
        if i == 0:
            model.add(Dense(units, activation=config.nn_activation, input_shape=(input_dim,)))
        else:
            model.add(Dense(units, activation=config.nn_activation))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    logger.info("Built neural network", layers=config.nn_layers, activation=config.nn_activation)
    return model

@memory.cache
def predict_nn(model: Sequential, X_scaled: np.ndarray) -> np.ndarray:
    """Cached neural network prediction."""
    return model.predict(X_scaled, batch_size=config.batch_size, verbose=0)  # config from import

def compute_shap_values(
    model, X: pd.DataFrame, sample_size: int, random_seed: int
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for model interpretability."""
    import shap
    rng = np.random.default_rng(random_seed)
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=rng.integers(0, 2**32))
    if isinstance(model, RandomForestRegressor):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample.values)
    elif isinstance(model, Sequential):
        explainer = shap.DeepExplainer(model, X_sample.values)
        shap_values = explainer.shap_values(X_sample.values)[0]
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim == 1:
        raise ValueError("SHAP values are a 1D vector; expected a 2D matrix")
    logger.info("Computed SHAP values", model_type=type(model).__name__, sample_size=len(X_sample))
    return shap_values, X_sample

def train_neural_network(X: pd.DataFrame, y: pd.Series, config: Config) -> Tuple[Sequential, StandardScaler]:
    """Train a neural network with cached validation splits and progress."""
    if len(X) < config.min_rows:
        raise MissingDataError(f"Dataset has {len(X)} rows, minimum is {config.min_rows}")
    if y.isna().any():
        raise ValueError("Target contains NaN values")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rng = np.random.default_rng(config.random_seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y.values, test_size=config.validation_split, random_state=rng.integers(0, 2**32)
    )
    
    model = build_neural_network(X_scaled.shape[1], config)
    early_stopping = RichProgressCallback(
        config.epochs, monitor='val_loss', patience=10, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(X_val, y_val),
        verbose=0,
        callbacks=[early_stopping]
    )
    logger.info("Trained neural network", final_loss=history.history['loss'][-1], final_val_loss=history.history['val_loss'][-1])
    return model, scaler
