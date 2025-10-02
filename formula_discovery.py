# formula_discovery.py
"""
Symbolic Formula Discovery Module.
Uses symbolic regression to evolve interpretable mathematical formulas from data.
Supports multiple methods: PySR (recommended for speed/diversity), gplearn (classic GP).
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict
import sympy as sp
from sklearn.metrics import r2_score

# Method 1: PySR
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    print("PySR not available; install with 'pip install pysr' for best performance.")

# Method 2: gplearn
try:
    from gplearn.genetic import SymbolicRegressor
    GPLEARN_AVAILABLE = True
except ImportError:
    GPLEARN_AVAILABLE = False
    print("gplearn not available; install with 'pip install gplearn'.")

class FormulaDiscoveryError(Exception):
    """Raised when formula discovery fails."""
    pass

def discover_formula(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    feature_names: Optional[List[str]] = None,
    method: str = "pysr",
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
        feature_names: Column names for symbolic output.
        method: "pysr" or "gplearn".
        max_complexity: Max equation complexity.
        n_iterations: Search steps.
        operators: List of unary/binary ops.
        target_name: Name for target in equation.
    
    Returns:
        Dict with 'equation', 'str_formula', 'score', 'complexity'.
    """
    # Convert inputs
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist() if feature_names is None else feature_names
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    # Ensure numpy arrays
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Default operators
    if operators is None:
        operators = ["add", "sub", "mul", "div", "pow", "exp", "log", "sin", "cos", "sqrt", "abs"]
    
    # Validate data
    if len(X) == 0 or len(y) == 0:
        raise FormulaDiscoveryError("Empty dataset provided")
    
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise FormulaDiscoveryError("Data contains NaN values")
    
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise FormulaDiscoveryError("Data contains infinite values")
    
    if method == "pysr" and PYSR_AVAILABLE:
        try:
            # PySR implementation
            binary_ops = [op for op in operators if op in ["add", "sub", "mul", "div", "pow"]]
            unary_ops = [op for op in operators if op in ["exp", "log", "sin", "cos", "sqrt", "abs", "square"]]
            
            model = PySRRegressor(
                niterations=n_iterations,
                binary_operators=binary_ops if binary_ops else ["add", "mul"],
                unary_operators=unary_ops if unary_ops else ["exp", "log"],
                maxsize=max_complexity,
                loss="loss(x, y) = (x - y)^2",
                model_selection="best",
                verbosity=0,
                progress=False
            )
            
            model.fit(X, y, variable_names=feature_names)
            
            # Get best equation
            if hasattr(model, 'sympy') and callable(model.sympy):
                equation = model.sympy()
            elif hasattr(model, 'get_best'):
                best_row = model.get_best()
                equation = sp.sympify(best_row['equation'])
            else:
                raise FormulaDiscoveryError("Could not extract equation from PySR model")
            
            # Calculate R² score
            y_pred = model.predict(X)
            score = r2_score(y, y_pred)
            
            complexity = len(list(sp.preorder_traversal(equation)))
            
        except Exception as e:
            raise FormulaDiscoveryError(f"PySR failed: {str(e)}")
    
    elif method == "gplearn" and GPLEARN_AVAILABLE:
        try:
            # gplearn implementation with compatibility fixes
            function_set = tuple([op for op in operators if op in 
                                 ("add", "sub", "mul", "div", "log", "sqrt", "sin", "cos")])
            if not function_set:
                function_set = ("add", "sub", "mul", "div")
            
            # Create model with explicit sklearn compatibility
            model = SymbolicRegressor(
                population_size=1000,
                generations=max(20, n_iterations // 10),
                tournament_size=20,
                stopping_criteria=0.01,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=0.9,
                verbose=0,
                parsimony_coefficient=0.01,
                function_set=function_set,
                random_state=42,
                n_jobs=1  # Single thread for compatibility
            )
            
            # Manual data validation to avoid _validate_data
            if X.shape[0] != y.shape[0]:
                raise FormulaDiscoveryError(f"X and y have different lengths: {X.shape[0]} vs {y.shape[0]}")
            
            # Fit the model
            model.fit(X, y)
            
            # Get predictions and calculate R²
            y_pred = model.predict(X)
            score = r2_score(y, y_pred)
            
            # Get program string
            program_str = str(model._program)
            
            # Replace variable names carefully
            for i in range(len(feature_names)):
                program_str = program_str.replace(f"X{i}", f"VAR_{i}_TEMP")
            for i, name in enumerate(feature_names):
                program_str = program_str.replace(f"VAR_{i}_TEMP", name)
            
            # Clean up common gplearn artifacts
            program_str = program_str.replace("add(", "(")
            program_str = program_str.replace("sub(", "(")
            program_str = program_str.replace("mul(", "(")
            program_str = program_str.replace("div(", "(")
            
            # Parse to SymPy
            try:
                equation = sp.sympify(program_str)
            except:
                # Fallback: create a simple representation
                equation = sp.Symbol("f(x)")
                program_str = str(model._program)
            
            complexity = model._program.length_
            
        except AttributeError as e:
            if "_validate_data" in str(e):
                raise FormulaDiscoveryError(
                    "gplearn compatibility issue detected. "
                    "Try: pip install --upgrade scikit-learn==1.0.2 gplearn==0.4.2"
                )
            else:
                raise FormulaDiscoveryError(f"gplearn failed: {str(e)}")
        except Exception as e:
            raise FormulaDiscoveryError(f"gplearn failed: {str(e)}")
    
    else:
        available = []
        if PYSR_AVAILABLE:
            available.append("pysr")
        if GPLEARN_AVAILABLE:
            available.append("gplearn")
        
        if not available:
            raise FormulaDiscoveryError(
                "No formula discovery methods available. "
                "Install with: pip install gplearn (or) pip install pysr"
            )
        else:
            raise FormulaDiscoveryError(
                f"Method '{method}' unavailable. Available: {', '.join(available)}"
            )
    
    # Simplify equation
    try:
        equation = sp.simplify(equation)
    except:
        pass  # Keep original if simplification fails
    
    # Create readable string with proper variable names
    str_formula = str(equation)
    for i, name in enumerate(feature_names):
        str_formula = str_formula.replace(f"x{i}", name)
    
    return {
        "equation": equation,
        "str_formula": str_formula,
        "score": float(score),
        "complexity": int(complexity),
        "feature_names": feature_names,
        "target_name": target_name
    }
