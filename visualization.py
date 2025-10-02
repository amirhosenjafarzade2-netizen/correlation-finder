# visualization.py
"""
Visualization module: All plot functions.
Returns figs with original variable names (from params).
Adapted for Streamlit: No display(), use st.plotly_chart/st.pyplot.
Preserves all graph types.
"""

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any  # Added Any
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import shap
from config import Config  # Added: For config in generate_viz
from utils import logger

def plot_heatmap(
    data: pd.DataFrame, 
    x_labels: List[str], 
    y_labels: List[str], 
    title: str, 
    x_title: str, 
    y_title: str
) -> go.Figure:
    """Generate an interactive heatmap with Plotly (uses original labels)."""
    if data.empty or data.isna().all().all() or not np.any(data.values != 0):
        logger.warning(f"Heatmap data for '{title}' is empty")
        return go.Figure()
    fig = px.imshow(
        data, x=x_labels, y=y_labels, title=title,
        labels={'x': x_title, 'y': y_title}, color_continuous_scale='Viridis',
        text_auto='.2f', width=800, height=800
    )
    fig.update_layout(title_font_size=16, xaxis_title_font_size=12, yaxis_title_font_size=12, font=dict(size=12))
    return fig

def plot_pairplot(df: pd.DataFrame, params: List[str], title: str) -> plt.Figure:
    """Generate a pair plot with seaborn (original param names)."""
    sns_plot = sns.pairplot(df[params], diag_kind='hist')
    fig = sns_plot.figure
    fig.suptitle(title, y=1.02, fontsize=16)
    return fig

def plot_parallel_coordinates(df: pd.DataFrame, params: List[str], title: str) -> go.Figure:
    """Generate parallel coordinates with Plotly."""
    fig = px.parallel_coordinates(
        df, dimensions=params, title=title,
        color=params[0], color_continuous_scale='Viridis'
    )
    fig.update_layout(title_font_size=16, width=800, height=600)
    return fig

def plot_boxplot(df: pd.DataFrame, params: List[str], title: str) -> go.Figure:
    """Generate box plot with Plotly."""
    fig = go.Figure()
    for param in params:
        fig.add_trace(go.Box(y=df[param], name=param))
    fig.update_layout(title=title, title_font_size=16, width=800, height=600, yaxis_title="Value")
    return fig

def plot_pca(df: pd.DataFrame, params: List[str], title: str) -> go.Figure:
    """2D PCA plot."""
    if len(params) < 2:
        logger.warning("Insufficient features for PCA")
        return go.Figure()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[params])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1], title=title,
        labels={'x': f'PC1 ({explained_variance[0]:.2%})', 'y': f'PC2 ({explained_variance[1]:.2%})'}
    )
    fig.update_layout(title_font_size=16, width=800, height=600)
    return fig

def plot_pca_3d(df: pd.DataFrame, params: List[str], title: str) -> go.Figure:
    """3D PCA plot."""
    if len(params) < 3:
        logger.warning("Insufficient features for 3D PCA")
        return go.Figure()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[params])
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    fig = px.scatter_3d(
        x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], title=title,
        labels={
            'x': f'PC1 ({explained_variance[0]:.2%})',
            'y': f'PC2 ({explained_variance[1]:.2%})',
            'z': f'PC3 ({explained_variance[2]:.2%})'
        }
    )
    fig.update_layout(title_font_size=16, width=800, height=600)
    return fig

def plot_rf_importance(rf_importance: Dict[str, pd.DataFrame], title: str) -> plt.Figure:
    """RF importance bar plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for target, importance_df in rf_importance.items():
        ax.bar(importance_df['Feature'], importance_df['Importance'], label=target, alpha=0.5)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance Score')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_shap_summary(shap_values: np.ndarray, X_sample: pd.DataFrame, title: str) -> plt.Figure:
    """SHAP summary plot."""
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar")
    plt.title(title, fontsize=16)
    return fig

def generate_viz_from_analysis(results: Dict[str, Any], params: List[str], config: Config) -> List[plt.Figure | go.Figure]:
    """Generate all selected visualizations using results from analysis."""
    figs = []
    selected_viz = config.visualizations
    
    if 'correlation' in selected_viz:
        fig = plot_heatmap(results['corr_matrix'], params, params, 'Spearman Correlation Heatmap', 'Parameters', 'Parameters')
        figs.append(fig)
    
    if 'mi' in selected_viz:
        fig = plot_heatmap(results['mi_matrix'], params, params, 'Normalized MI Heatmap', 'Parameters', 'Parameters')
        figs.append(fig)
    
    if 'mi_saturated' in selected_viz and not results['mi_saturated'].isna().all().all():
        fig = plot_heatmap(results['mi_saturated'], params, params, 'Normalized MI Heatmap (Saturated)', 'Parameters', 'Parameters')
        figs.append(fig)
    
    if 'mi_undersaturated' in selected_viz and not results['mi_undersaturated'].isna().all().all():
        fig = plot_heatmap(results['mi_undersaturated'], params, params, 'Normalized MI Heatmap (Undersaturated)', 'Parameters', 'Parameters')
        figs.append(fig)
    
    if 'pairplot' in selected_viz:
        fig = plot_pairplot(results['df'], params, 'Pair Plot of Parameters')  # Use 'df' from results
        figs.append(fig)
    
    if 'parallel' in selected_viz:
        fig = plot_parallel_coordinates(results['df'], params, 'Parallel Coordinates')
        figs.append(fig)
    
    if 'boxplot' in selected_viz:
        fig = plot_boxplot(results['df'], params, 'Box Plot of Parameters')
        figs.append(fig)
    
    if 'pca' in selected_viz:
        fig = plot_pca(results['df'], params, 'PCA 2D Plot')
        figs.append(fig)
    
    if 'pca_3d' in selected_viz:
        fig = plot_pca_3d(results['df'], params, 'PCA 3D Plot')
        figs.append(fig)
    
    if 'rf_importance' in selected_viz and results['rf_importance']:
        fig = plot_rf_importance(results['rf_importance'], 'RF Feature Importance')
        figs.append(fig)
    
    if 'shap' in selected_viz and results['shap_values_dict']:
        for target, (shap_values, X_sample) in results['shap_values_dict'].items():
            fig = plot_shap_summary(shap_values, X_sample, f'SHAP Summary for {target}')
            figs.append(fig)
    
    return figs
