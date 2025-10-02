# config.py
"""
Configuration module for the Data Analysis Dashboard.
Defines the Pydantic Config model and defaults.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List

class Config(BaseModel):
    """Configuration model with validation."""
    random_seed: int = 42
    min_rows: int = Field(default=10, gt=0)
    epochs: int = Field(default=50, gt=0)
    validation_split: float = Field(default=0.2, ge=0.0, le=1.0)
    bubble_point_default: float = 2000
    pop_size: int = Field(default=50, gt=0)
    n_generations: int = Field(default=50, gt=0)
    n_calls: int = Field(default=50, gt=0)
    visualizations: List[str] = Field(default_factory=lambda: [
        "correlation", "mi", "mi_saturated", "mi_undersaturated", "pairplot", 
        "parallel", "boxplot", "pca", "pca_3d", "rf_importance", "shap"
    ])
    nn_layers: List[int] = Field(default_factory=lambda: [64, 32])
    nn_activation: str = "relu"
    batch_size: int = Field(default=32, gt=0)
    n_jobs: int = Field(default=-1, ge=-1)
    shap_sample_size: int = Field(default=100, gt=0)

    @field_validator('visualizations')
    @classmethod
    def validate_visualizations(cls, v):
        valid = ["correlation", "mi", "mi_saturated", "mi_undersaturated", "pairplot", 
                 "parallel", "boxplot", "pca", "pca_3d", "rf_importance", "shap"]
        if not all(x in valid for x in v):
            raise ValueError(f"Invalid visualization types: {set(v) - set(valid)}")
        return v

# Default config for easy instantiation
DEFAULT_CONFIG = {
    "random_seed": 42,
    "min_rows": 10,
    "epochs": 50,
    "validation_split": 0.2,
    "bubble_point_default": 2000,
    "pop_size": 50,
    "n_generations": 50,
    "n_calls": 50,
    "visualizations": ["correlation", "mi", "mi_saturated", "mi_undersaturated", "pairplot", "parallel", "boxplot", "pca", "pca_3d", "rf_importance", "shap"],
    "nn_layers": [64, 32],
    "nn_activation": "relu",
    "batch_size": 32,
    "n_jobs": -1,
    "shap_sample_size": 100
}

# Pressure column candidates (hardcoded as per user: only 'P' and 'pressure')
PRESSURE_COLS = ['P', 'pressure']
