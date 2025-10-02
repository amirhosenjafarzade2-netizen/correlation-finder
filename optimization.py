# optimization.py
"""
Optimization module: GA, Bayesian, integrated target opt.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from deap import base, creator, tools
from skopt import gp_minimize
from skopt.space import Real
from config import Config
from utils import memory, logger
from models import predict_nn, Sequential, StandardScaler, train_neural_network  # Added imports

def setup_genetic_algorithm(creator_class_name: str, fitness_weights: tuple[float, ...]) -> None:
    """Set up genetic algorithm classes."""
    if not hasattr(creator, creator_class_name):
        creator.create(creator_class_name, base.Fitness, weights=fitness_weights)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=getattr(creator, creator_class_name))
    logger.info("Set up genetic algorithm", class_name=creator_class_name)

@memory.cache
def evaluate_individual_target(individual: tuple[float, ...], model: Sequential, scaler: StandardScaler, feature_columns: list[str], batch_size: int = 32) -> float:
    """Cached evaluation for genetic algorithm in optimize_target."""
    X_test = pd.DataFrame([individual], columns=feature_columns)
    X_test_scaled = scaler.transform(X_test)
    return predict_nn(model, X_test_scaled, batch_size)[0][0]

def optimize_target_ga(
    df: pd.DataFrame, 
    params: List[str], 
    target_name: str, 
    model: Sequential, 
    scaler: StandardScaler, 
    config: Config
) -> Tuple[pd.DataFrame, List[plt.Figure]]:
    """Optimize target using genetic algorithm."""
    features = df[params].drop(columns=[target_name])
    target = df[target_name]
    target_mean = target.mean()
    target_std = target.std()
    if target_std == 0:
        raise ValueError("Target std is 0")
    y_scaled = (target - target_mean) / target_std
    
    setup_genetic_algorithm("FitnessMax", (1.0,))
    
    rng = np.random.default_rng(config.random_seed)
    
    def create_individual() -> creator.Individual:
        return creator.Individual([rng.uniform(df[p].min(), df[p].max()) for p in features.columns])
    
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual_target, model=model, scaler=scaler, feature_columns=features.columns, batch_size=config.batch_size)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    population = toolbox.population(n=config.pop_size)
    
    figs: List[plt.Figure] = []
    best_fitness_history = []
    for gen in range(config.n_generations):
        offspring = toolbox.select(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < 0.7:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values
        for ind in offspring:
            if rng.random() < 0.2:
                toolbox.mutate(ind)
                for i in range(len(ind)):
                    ind[i] = max(df[features.columns[i]].min(), min(df[features.columns[i]].max(), ind[i]))
                del ind.fitness.values
        fitnesses = [toolbox.evaluate(ind) for ind in offspring]
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = (fit,)
        population[:] = offspring
        best_ind = tools.selBest(population, 1)[0]
        best_fitness_history.append(best_ind.fitness.values[0])
    
    best_ind = tools.selBest(population, 1)[0]
    
    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(1, config.n_generations + 1), best_fitness_history, marker='o')
    plt.title(f'Optimization Progress for {target_name} (GA)')
    plt.xlabel('Generation')
    plt.ylabel(f'Predicted {target_name} (Scaled)')
    plt.grid(True)
    figs.append(fig)
    
    optimal_params = best_ind
    optimal_df = pd.DataFrame([optimal_params], columns=features.columns)
    scaled_pred = predict_nn(model, scaler.transform(optimal_df), config.batch_size)[0][0]
    optimal_df[target_name] = scaled_pred * target_std + target_mean
    
    return optimal_df, figs

def optimize_target_bayesian(
    df: pd.DataFrame, 
    params: List[str], 
    target_name: str, 
    model: Sequential, 
    scaler: StandardScaler, 
    config: Config
) -> Tuple[pd.DataFrame, List[plt.Figure]]:
    """Optimize target using Bayesian optimization."""
    features = df[params].drop(columns=[target_name])
    target = df[target_name]
    target_mean = target.mean()
    target_std = target.std()
    if target_std == 0:
        raise ValueError("Target std is 0")
    
    search_space = [Real(df[p].min(), df[p].max(), name=p) for p in features.columns]
    
    @memory.cache
    def objective(params_list: List[float]) -> float:
        X_test = pd.DataFrame([params_list], columns=features.columns)
        X_test_scaled = scaler.transform(X_test)
        scaled_pred = predict_nn(model, X_test_scaled, config.batch_size)[0][0]
        return -(scaled_pred)  # Negative for maximization
    
    figs: List[plt.Figure] = []
    rng = np.random.default_rng(config.random_seed)
    res = gp_minimize(
        objective,
        search_space,
        n_calls=config.n_calls,
        random_state=rng.integers(0, 2**32)
    )
    
    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(res.func_vals) + 1), -np.array(res.func_vals), marker='o')
    plt.title(f'Optimization Progress for {target_name} (Bayesian)')
    plt.xlabel('Iteration')
    plt.ylabel(f'Predicted {target_name} (Scaled)')
    plt.grid(True)
    figs.append(fig)
    
    optimal_params = res.x
    optimal_df = pd.DataFrame([optimal_params], columns=features.columns)
    scaled_pred = -res.fun
    optimal_df[target_name] = scaled_pred * target_std + target_mean
    
    return optimal_df, figs

def optimize_target(
    df: pd.DataFrame, 
    params: List[str], 
    target_name: str, 
    config: Config, 
    optimizer: str = 'ga'
) -> Tuple[pd.DataFrame, List[plt.Figure], Tuple[np.ndarray, pd.DataFrame] | None]:
    """Optimize a target parameter using NN and specified optimizer."""
    if target_name not in params:
        raise ValueError(f"Target '{target_name}' not in parameters")
    
    features = df[params].drop(columns=[target_name])
    target = df[target_name]
    target_mean = target.mean()
    target_std = target.std()
    if target_std == 0:
        raise ValueError("Target std is 0")
    y_scaled = (target - target_mean) / target_std
    
    model, scaler = train_neural_network(features, y_scaled, config)  # Pass scaled y
    
    # NN importance
    weights = model.layers[0].get_weights()[0]
    feature_importance = np.mean(np.abs(weights), axis=1)
    importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importance}).sort_values('Importance', ascending=False)
    
    figs = []
    fig_nn = plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.title(f'NN Feature Importance for {target_name}')
    plt.xticks(rotation=45)
    figs.append(fig_nn)
    
    shap_values, X_sample = None, None
    if 'shap' in config.visualizations:
        try:
            shap_values, X_sample = compute_shap_values(model, features, config.shap_sample_size, config.random_seed)
        except Exception as e:
            logger.error(f"Failed SHAP for NN", error=str(e))
    
    if optimizer == 'ga':
        optimal_df, opt_figs = optimize_target_ga(df, params, target_name, model, scaler, config)
    else:
        optimal_df, opt_figs = optimize_target_bayesian(df, params, target_name, model, scaler, config)
    
    figs.extend(opt_figs)
    return optimal_df, figs, (shap_values, X_sample) if shap_values is not None else None
