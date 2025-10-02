# formula_discovery.py
"""
Symbolic Formula Discovery Module.
Uses symbolic regression to evolve interpretable mathematical formulas from data.
Supports gplearn (primary for Streamlit Cloud compatibility).
PySR disabled due to permission issues on Cloud.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict
import sympy as sp
from sklearn.metrics import r2_score

# Skip PySR due to Julia permission errors on Streamlit Cloud
PYSR_AVAILABLE = False

# gplearn (works with sklearn 1.1.x)
GPLEARN_AVAILABLE = False
try:
    from gplearn.genetic import SymbolicRegressor
    GPLEARN_AVAILABLE = True
except ImportError:
    print("gplearn not available; install with 'pip install gplearn==0.4.2'.")

class FormulaDiscoveryError(Exception):
    """Raised when formula discovery fails."""
    pass

def discover_formula(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    feature_names: Optional[List[str]] = None,
    method: str = "gplearn",  # Only gplearn supported
    max_complexity: int = 10,
    n_iterations: int = 100,
    operators: Optional[List[str]] = None,
    target_name: str = "y"
) -> Dict[str, any]:
    """
    Discover a symbolic formula using gplearn.
    
    Args:
        X: Features (n_samples, n_features).
        y: Target (n_samples,).
        feature_names: Column names for symbolic output.
        method: Must be "gplearn".
        max_complexity: Max equation complexity.
        n_iterations: Search steps.
        operators: List of unary/binary ops.
        target_name: Name for target in equation.
    
    Returns:
        Dict with 'equation', 'str_formula', 'score', 'complexity'.
    """
    if method != "gplearn":
        raise FormulaDiscoveryError(f"Method '{method}' not supported. Use 'gplearn'.")

    if not GPLEARN_AVAILABLE:
        raise FormulaDiscoveryError(
            "gplearn not available. Add 'gplearn==0.4.2' to requirements.txt and pin 'scikit-learn==1.1.3'."
        )

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
        operators = ["add", "sub", "mul", "div", "log", "sqrt", "sin", "cos"]
    
    # Validate data
    if len(X) == 0 or len(y) == 0:
        raise FormulaDiscoveryError("Empty dataset provided")
    
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise FormulaDiscoveryError("Data contains NaN values")
    
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise FormulaDiscoveryError("Data contains infinite values")

    try:
        # gplearn implementation
        function_set = tuple([op for op in operators if op in 
                             ("add", "sub", "mul", "div", "log", "sqrt", "sin", "cos")])
        if not function_set:
            function_set = ("add", "sub", "mul", "div")
        
        # Limit generations based on n_iterations for performance
        generations = min(max(5, n_iterations // 20), 50)  # Cap for Cloud
        
        model = SymbolicRegressor(
            population_size=500,  # Reduced for speed
            generations=generations,
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
            n_jobs=1,
            const_range=(-1.0, 1.0),
            init_depth=(2, 6),
            metric="spearman"  # Use Spearman for robustness
        )
        
        # Fit the model
        model.fit(X, y)
        
        # Get predictions and calculate R²
        y_pred = model.predict(X)
        score = r2_score(y, y_pred)
        
        # Get program string
        program_str = str(model._program)
        
        # Replace variable names
        for i, name in enumerate(feature_names):
            program_str = program_str.replace(f"X{i}", name)
        
        # Clean up gplearn syntax
        program_str = program_str.replace("add(", "(").replace("sub(", "(").replace("mul(", "(").replace("div(", "(")
        
        # Parse to SymPy
        try:
            equation = sp.sympify(program_str)
        except Exception:
            equation = sp.Symbol("f(x)")
            program_str = str(model._program)  # Fallback
        
        complexity = model._program.length_
        
    except Exception as e:
        raise FormulaDiscoveryError(f"gplearn failed: {str(e)}")
    
    # Simplify equation
    try:
        equation = sp.simplify(equation)
    except:
        pass
    
    # Create readable string
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
