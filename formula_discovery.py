# formula_discovery.py
"""
Symbolic Formula Discovery Module.
Uses symbolic regression to evolve interpretable mathematical formulas from data.
Supports multiple methods: PySR (recommended for speed/diversity), gplearn (classic GP).
Customizable operators for exponential, logarithmic, trigonometric, arithmetic, power, etc.
Integrate into Streamlit app via: formula = discover_formula(df[features], df[target], method="gplearn")
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict
import sympy as sp  # For equation simplification/evaluation
from sklearn.metrics import r2_score  # For manual R² computation to avoid compat issues
import structlog

# Setup logging
logger = structlog.get_logger()

# Method 1: PySR (GPU-accelerated GP; best for large data/diverse ops)
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True  # Enable locally; set to False for Streamlit Cloud
except ImportError:
    PYSR_AVAILABLE = False
    logger.warning("PySR not available; install with 'pip install pysr' and ensure Julia is installed.")

# Method 2: gplearn (Pure Python GP; reliable fallback)
try:
    from gplearn.genetic import SymbolicRegressor
    from sklearn.utils.validation import check_array  # Explicit import for validation
    GPLEARN_AVAILABLE = True
except ImportError:
    GPLEARN_AVAILABLE = False
    logger.warning("gplearn not available; install with 'pip install gplearn'.")

class FormulaDiscoveryError(Exception):
    """Raised when formula discovery fails."""
    pass

def discover_formula(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    feature_names: Optional[List[str]] = None,
    method: str = "gplearn",  # Default to gplearn
    max_complexity: int = 10,  # Pareto front limit for equation complexity
    n_iterations: int = 100,  # Search iterations/generations
    operators: Optional[List[str]] = None,  # Custom ops; default includes all requested
    target_name: str = "y"
) -> Dict[str, any]:
    """
    Discover a symbolic formula using the selected method.
    
    Args:
        X: Features (n_samples, n_features).
        y: Target (n_samples,).
        feature_names: Column names for symbolic output (e.g., ['Bo', 'Rs']).
        method: "pysr" (recommended; fast, diverse ops) or "gplearn" (classic GP).
        max_complexity: Max equation complexity (balance fit/simplicity).
        n_iterations: Search steps (higher = better but slower).
        operators: List of unary/binary ops (e.g., ["exp", "log", "sin", "add"]).
        target_name: Name for target in equation (e.g., "Rs").
    
    Returns:
        Dict with 'equation' (SymPy expr), 'str_formula' (readable string),
        'score' (fit metric, e.g., R²), 'complexity' (op count).
    
    Raises:
        FormulaDiscoveryError: If method unavailable or discovery fails.
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist() if feature_names is None else feature_names
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    # Ensure data is validated (fixes _validate_data issues with sklearn compat)
    try:
        X = check_array(X, ensure_2d=True, dtype=np.float64, force_all_finite=True)
        y = np.asarray(y, dtype=np.float64).ravel()
    except Exception as e:
        logger.error("Data validation failed", error=str(e))
        raise FormulaDiscoveryError(f"Data validation failed: {e}")
    
    # Default operators: arithmetic, power, exp/log, trig
    if operators is None:
        operators = [
            # Binary
            "add", "sub", "mul", "div", "pow",
            # Unary
            "exp", "log", "sin", "cos", "sqrt", "abs"
        ]
    
    if method == "pysr" and PYSR_AVAILABLE:
        try:
            # PySR: Best for diverse ops; Pareto-optimal equations
            model = PySRRegressor(
                niterations=n_iterations,
                binary_operators=operators[:5],  # Arithmetic/power
                unary_operators=operators[5:],  # Exp/log/trig
                maxsize=max_complexity,
                loss="mse",  # Or "r2" for correlation focus
                model_selection="pareto"  # Balance fit/complexity
            )
            model.fit(X, y)
            best_eq = model.sympy()[-1]  # Best equation from Pareto front
            equation = sp.sympify(best_eq)
            score = model.equations_[-1][2]  # Loss (lower better; convert to R² if needed)
            complexity = len(sp.preorder_traversal(equation))
            str_formula = sp.pretty(equation, use_unicode=True)
            logger.info("PySR formula discovered", str_formula=str_formula, score=score, complexity=complexity)
        except Exception as e:
            logger.error("PySR failed", error=str(e))
            raise FormulaDiscoveryError(f"PySR failed: {e}")
    
    elif method == "gplearn" and GPLEARN_AVAILABLE:
        try:
            # gplearn: Customizable GP; good fallback
            model = SymbolicRegressor(
                population_size=1000,  # Smaller for faster runs
                generations=10,  # Fixed small number to avoid long runs/errors
                tournament_size=20,
                stopping_criteria=0.01,  # Stop if fit improves <1%
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=0.9,
                verbose=0,  # Silent for Streamlit
                parsimony_coefficient=0.01,  # Penalize complexity
                function_set=("add", "sub", "mul", "div", "log", "exp", "sin", "cos", "sqrt"),
                random_state=None,  # Avoid deprecation
                n_jobs=1  # Safe default for compatibility
            )
            # Explicit kwargs for sklearn 1.5+ compatibility
            model.fit(X=X, y=y)
            equation_str = model._program[1]  # Raw program string
            # Adapt variables for multi-features
            for i, name in enumerate(feature_names):
                equation_str = equation_str.replace(f'X{i}', name)
            equation = sp.sympify(equation_str)
            # Manual R² to avoid issues
            y_pred = model.predict(X=X)
            score = r2_score(y, y_pred)
            complexity = model._program[0]  # Depth as proxy
            str_formula = sp.pretty(equation, use_unicode=True)
            logger.info("gplearn formula discovered", str_formula=str_formula, score=score, complexity=complexity)
        except Exception as e:
            logger.error("gplearn failed", error=str(e))
            raise FormulaDiscoveryError(f"gplearn failed: {e}")
    
    else:
        # Fallback: Force gplearn if available
        if GPLEARN_AVAILABLE:
            logger.info("Falling back to gplearn", requested_method=method)
            return discover_formula(X, y, feature_names, "gplearn", max_complexity, n_iterations, operators, target_name)
        else:
            logger.error("No formula discovery methods available", requested_method=method)
            raise FormulaDiscoveryError(f"Method '{method}' unavailable. Install gplearn or pysr.")
    
    # Simplify with SymPy (common to both)
    try:
        equation = sp.simplify(equation)
        str_formula = str(equation)
    except Exception as e:
        logger.warning("SymPy simplification failed; using raw equation", error=str(e))
    
    return {
        "equation": equation,  # SymPy expr for eval/plot
        "str_formula": str_formula,  # Readable string
        "score": score,  # Fit metric
        "complexity": complexity,
        "feature_names": feature_names,
        "target_name": target_name
    }

# Example usage (for testing/integration)
if __name__ == "__main__":
    # Sample data
    X_sample = pd.DataFrame({
        'x0': np.random.rand(100),
        'x1': np.random.rand(100),
        'x2': np.random.rand(100)
    })
    y_sample = np.exp(X_sample['x0']) + np.sin(X_sample['x1']) + X_sample['x2'] + np.random.normal(0, 0.1, 100)
    
    try:
        formula = discover_formula(X_sample, y_sample, feature_names=['x0', 'x1', 'x2'], method="gplearn")
        print(f"Discovered: {formula['str_formula']}")
        print(f"Score (R²): {formula['score']:.4f}")
    except FormulaDiscoveryError as e:
        print(f"Error: {e}")
