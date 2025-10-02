# formula_runner.py
import sys
import json
import argparse
from typing import Dict, Any
import pandas as pd
import numpy as np
from formula_discovery import discover_formula  # Assumes this is in your path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, required=True, help='JSON list of feature names')
    parser.add_argument('--data_file', type=str, required=True, help='Path to temp CSV with X and y')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--method', type=str, default='gplearn', help='Method: gplearn')
    parser.add_argument('--max_complexity', type=int, default=10)
    parser.add_argument('--n_iterations', type=int, default=100)
    args = parser.parse_args()

    # Load data from temp CSV (X features + y target)
    df = pd.read_csv(args.data_file)
    X = df[json.loads(args.features)]  # Features
    y = df[args.target]

    # Run discovery
    result = discover_formula(
        X, y,
        feature_names=json.loads(args.features),
        method=args.method,
        max_complexity=args.max_complexity,
        n_iterations=args.n_iterations,
        target_name=args.target
    )

    # Output as JSON
    print(json.dumps(result))  # stdout for easy capture

if __name__ == '__main__':
    main()
