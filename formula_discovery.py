# formula_discovery.py
"""
Symbolic Formula Discovery Module.
Uses symbolic regression to evolve interpretable mathematical formulas from data.
Supports multiple methods: PySR (recommended for speed/diversity), gplearn (classic GP), linear (cloud fallback).
Customizable operators for exponential, logarithmic, trigonometric, arithmetic, power, etc.
Integrate into Streamlit app via: formula = discover_formula(df[features], df[target], method="gplearn")
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict
import sympy as sp
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array
from sklearn.linear_model import LinearRegression  # Fallback for cloud
import structlog

logger = structlog.get_logger()

# Method 1: PySR (GPU-accelerated; local only)
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = False  # Set False for Streamlit Cloud; True for local with Julia
except ImportError:
    PYSR_AVAILABLE = False
    logger.warning("PySR not available; install with 'pip install pysr' and ensure Julia is installed.")

# Method 2: gplearn (Pure Python GP)
try:
    from gplearn.genetic import SymbolicRegressor
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
    method: str = "gplearn",  # Default to gplearn for local; linear for cloud
    max_complexity: int = 10,
    n_iterations: int = 100,
    operators: Optional[List[str]] = None,
    target_name: str = "y"
) -> Dict[str, any]:
    """
    Discover a symbolic formula using the selected method.
    
    Args:
        X: Features (n_samples, n_features).
        y: Target (n_samples,).
        feature_names: Column names for symbolic output (e.g., ['Bo', 'Rs']).
        method: "pysr" (fast, diverse), "gplearn" (classic GP), or "linear" (cloud fallback).
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
    
    try:
        X = check_array(X, ensure_2d=True, dtype=np.float64, force_all_finite=True)
        y = np.asarray(y, dtype=np.float64).ravel()
    except Exception as e:
        logger.error("Data validation failed", error=str(e))
        raise FormulaDiscoveryError(f"Data validation failed: {e}")
    
    if operators is None:
        operators = ["add", "sub", "mul", "div", "pow", "exp", "log", "sin", "cos", "sqrt", "abs"]
    
    equation = None
    score = 0.0
    complexity = 0
    str_formula = ""

    if method == "pysr" and PYSR_AVAILABLE:
        try:
            model = PySRRegressor(
                niterations=n_iterations,
                binary_operators=operators[:5],
                unary_operators=operators[5:],
                maxsize=max_complexity,
                loss="mse",
                model_selection="pareto"
            )
            model.fit(X, y)
            best_eq = model.sympy()[-1]
            equation = sp.sympify(best_eq)
            score = 1 - model.equations_[-1][2] / np.var(y)  # Approx R² from MSE
            complexity = len(sp.preorder_traversal(equation))
            str_formula = sp.pretty(equation, use_unicode=True)
            logger.info("PySR formula discovered", str_formula=str_formula, score=score, complexity=complexity)
        except Exception as e:
            logger.error("PySR failed", error=str(e))
            raise FormulaDiscoveryError(f"PySR failed: {e}")
    
    elif method == "gplearn" and GPLEARN_AVAILABLE:
        try:
            model = SymbolicRegressor(
                population_size=1000,
                generations=10,
                tournament_size=20,
                stopping_criteria=0.01,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=0.9,
                verbose=0,
                parsimony_coefficient=0.01,
                function_set=("add", "sub", "mul", "div", "log", "exp", "sin", "cos", "sqrt"),
                random_state=None,
                n_jobs=1
            )
            model.fit(X=X, y=y)
            equation_str = model._program[1]
            for i, name in enumerate(feature_names):
                equation_str = equation_str.replace(f'X{i}', name)
            equation = sp.sympify(equation_str)
            y_pred = model.predict(X=X)
            score = r2_score(y, y_pred)
            complexity = model._program[0]
            str_formula = sp.pretty(equation, use_unicode=True)
            logger.info("gplearn formula discovered", str_formula=str_formula, score=score, complexity=complexity)
        except Exception as e:
            logger.error("gplearn failed", error=str(e))
            raise FormulaDiscoveryError(f"gplearn failed: {e}")
    
    elif method == "linear":
        try:
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            score = r2_score(y, y_pred)
            coefs = model.coef_
            intercept = model.intercept_
            terms = [sp.Float(intercept)]
            for coef, fname in zip(coefs, feature_names):
                terms.append(sp.Float(coef) * sp.symbols(fname))
            equation = sp.Add(*terms)
            complexity = len(feature_names) + 1
            str_formula = sp.pretty(equation, use_unicode=True)
            logger.info("Linear regression formula discovered", str_formula=str_formula, score=score)
        except Exception as e:
            logger.error("Linear regression failed", error=str(e))
            raise FormulaDiscoveryError(f"Linear regression failed: {e}")
    
    else:
        if GPLEARN_AVAILABLE:
            logger.info("Falling back to gplearn", requested_method=method)
            return discover_formula(X, y, feature_names, "gplearn", max_complexity, n_iterations, operators, target_name)
        elif PYSR_AVAILABLE:
            logger.info("Falling back to pysr", requested_method=method)
            return discover_formula(X, y, feature_names, "pysr", max_complexity, n_iterations, operators, target_name)
        else:
            logger.info("Falling back to linear regression", requested_method=method)
            return discover_formula(X, y, feature_names, "linear", max_complexity, n_iterations, operators, target_name)
    
    try:
        equation = sp.simplify(equation)
        str_formula = str(equation)
    except Exception as e:
        logger.warning("SymPy simplification failed; using raw equation", error=str(e))
    
    return {
        "equation": equation,
        "str_formula": str_formula,
        "score": score,
        "complexity": complexity,
        "feature_names": feature_names,
        "target_name": target_name
    }

# Example usage (for testing/integration)
if __name__ == "__main__":
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
