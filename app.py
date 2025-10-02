# app.py
"""
Main Streamlit app: UI, orchestration, display.
Integrates all modules; handles inputs, runs tasks, shows results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.io as pio
from typing import List
from config import Config, DEFAULT_CONFIG, PRESSURE_COLS
from data import load_and_preprocess_data, detect_pressure_column, classify_regimes
from analysis import analyze_relationships
from models import train_neural_network, compute_shap_values
from optimization import optimize_target
from visualization import generate_viz_from_analysis
from utils import logger, MissingDataError, InvalidPressureError, impute_missing_values

# Configure Plotly for Streamlit
pio.renderers.default = 'browser'  # Or 'notebook' if needed

st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

@st.cache_data
def get_config_from_ui() -> Config:
    """Create Config from sidebar inputs."""
    st.sidebar.header("Configuration")
    random_seed = st.sidebar.number_input("Random Seed", value=DEFAULT_CONFIG["random_seed"])
    min_rows = st.sidebar.number_input("Min Rows", value=DEFAULT_CONFIG["min_rows"], min_value=1)
    epochs = st.sidebar.number_input("NN Epochs", value=DEFAULT_CONFIG["epochs"], min_value=1)
    validation_split = st.sidebar.slider("Validation Split", 0.0, 1.0, DEFAULT_CONFIG["validation_split"])
    pop_size = st.sidebar.number_input("GA Pop Size", value=DEFAULT_CONFIG["pop_size"], min_value=1)
    n_generations = st.sidebar.number_input("GA Generations", value=DEFAULT_CONFIG["n_generations"], min_value=1)
    n_calls = st.sidebar.number_input("Bayesian Calls", value=DEFAULT_CONFIG["n_calls"], min_value=1)
    nn_layers = st.sidebar.text_input("NN Layers", value=",".join(map(str, DEFAULT_CONFIG["nn_layers"]))).split(",")
    nn_layers = [int(x.strip()) for x in nn_layers]
    batch_size = st.sidebar.number_input("Batch Size", value=DEFAULT_CONFIG["batch_size"], min_value=1)
    shap_sample_size = st.sidebar.number_input("SHAP Sample Size", value=DEFAULT_CONFIG["shap_sample_size"], min_value=1)
    
    # Viz selection
    default_viz = DEFAULT_CONFIG["visualizations"]
    visualizations = st.sidebar.multiselect("Visualizations", options=[
        "correlation", "mi", "mi_saturated", "mi_undersaturated", "pairplot", 
        "parallel", "boxplot", "pca", "pca_3d", "rf_importance", "shap"
    ], default=default_viz)
    
    config_dict = {
        "random_seed": random_seed,
        "min_rows": min_rows,
        "epochs": epochs,
        "validation_split": validation_split,
        "bubble_point_default": DEFAULT_CONFIG["bubble_point_default"],
        "pop_size": pop_size,
        "n_generations": n_generations,
        "n_calls": n_calls,
        "visualizations": visualizations,
        "nn_layers": nn_layers,
        "nn_activation": DEFAULT_CONFIG["nn_activation"],
        "batch_size": batch_size,
        "n_jobs": DEFAULT_CONFIG["n_jobs"],
        "shap_sample_size": shap_sample_size
    }
    return Config(**config_dict)

def main():
    st.title("Data Analysis Dashboard")
    
    config = get_config_from_ui()
    rng = np.random.default_rng(config.random_seed)
    
    # File upload
    uploaded_files = st.file_uploader("Upload Excel files", accept_multiple_files=True, type=['xlsx', 'xls'])
    
    # Row sampling
    n_rows = st.number_input("Number of rows to sample (0 for all)", min_value=0, value=0)
    if n_rows == 0:
        n_rows = None
    
    if st.button("Load Data"):
        try:
            df, params = load_and_preprocess_data(uploaded_files, n_rows, config, rng)
            st.session_state.df = df
            st.session_state.params = params
            st.success(f"Loaded {len(df)} rows with {len(params)} parameters: {', '.join(params)}")
            
            # Data preview
            with st.expander("Data Preview"):
                st.dataframe(df.head())
                st.metric("Missing % (post-impute)", 0)  # Placeholder; compute if needed
            
            # Pressure detection
            pressure_col = detect_pressure_column(params)
            if pressure_col:
                st.info(f"Detected pressure column: {pressure_col}")
                bubble_point = st.number_input("Bubble Point Pressure", value=config.bubble_point_default)
                st.session_state.pressure_col = pressure_col
                st.session_state.bubble_point = bubble_point
            else:
                st.info("No pressure column ('P' or 'pressure') detected; skipping regime split.")
                st.session_state.pressure_col = None
                st.session_state.bubble_point = None
            
        except Exception as e:
            st.error(f"Data loading error: {str(e)}")
            return
    
    if 'df' not in st.session_state:
        st.warning("Please load data first.")
        return
    
    df = st.session_state.df
    params = st.session_state.params
    pressure_col = st.session_state.get('pressure_col')
    bubble_point = st.session_state.get('bubble_point')
    
    # Task selection
    task = st.selectbox("Task", ["1: Relationship Analysis", "2: Target Optimization", "3: Both"])
    save_files = st.checkbox("Save/Download results")
    
    if task in ["2", "3"]:
        optimizer = st.selectbox("Optimizer", ["ga", "bayesian"])
        target_name = st.selectbox("Target Parameter", params)
        interactive = st.checkbox("Enable Interactive Exploration")
    
    all_figures = []
    
    if st.button("Run Analysis") or task == "3":
        try:
            if pressure_col:
                df_classified = classify_regimes(df.copy(), pressure_col, bubble_point)
            else:
                df_classified = df.copy()
            
            results = analyze_relationships(df_classified, params, pressure_col, bubble_point, config)
            figs = generate_viz_from_analysis(results, params, config)
            all_figures.extend(figs)
            
            # Display figs
            for i, fig in enumerate(figs):
                if isinstance(fig, go.Figure):
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.pyplot(fig)
            
            # Downloads if save_files
            if save_files:
                # Example: Download corr_matrix
                csv = results['corr_matrix'].to_csv(index=True)
                st.download_button("Download Correlation Matrix", csv, "correlation.csv", "text/csv")
                # Add more...
            
            st.success("Analysis complete!")
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
    
    if task in ["2", "3"] and st.button("Run Optimization"):
        try:
            optimal_df, opt_figs, shap_data = optimize_target(df, params, target_name, config, optimizer)
            all_figures.extend(opt_figs)
            
            st.dataframe(optimal_df)
            for fig in opt_figs:
                st.pyplot(fig)
            
            if save_files:
                excel = optimal_df.to_excel(index=False)
                st.download_button("Download Optimal Params", excel, f"optimal_{target_name}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            if interactive:
                # Interactive sliders (replaces ipywidgets)
                st.subheader("Interactive Parameter Exploration")
                features = [p for p in params if p != target_name]
                model, scaler = train_neural_network(df[features], df[target_name], config)
                col1, col2 = st.columns(2)
                test_values = {}
                for i, feat in enumerate(features):
                    with col1 if i % 2 == 0 else col2:
                        test_values[feat] = st.slider(feat, df[feat].min(), df[feat].max(), df[feat].mean())
                
                X_test = pd.DataFrame([test_values], columns=features)
                X_test_scaled = scaler.transform(X_test)
                pred = predict_nn(model, X_test_scaled)[0][0]
                st.metric("Predicted Value", pred)
                
                # Simple sensitivity plot (fix bug: plot vs one var)
                selected_feat = st.selectbox("Sensitivity to", features)
                feat_range = np.linspace(df[selected_feat].min(), df[selected_feat].max(), 50)
                sens_preds = []
                for val in feat_range:
                    temp_df = X_test.copy()
                    temp_df[selected_feat] = val
                    temp_scaled = scaler.transform(temp_df)
                    sens_preds.append(predict_nn(model, temp_scaled)[0][0])
                fig_sens = px.line(x=feat_range, y=sens_preds, title=f"Sensitivity of {target_name} to {selected_feat}")
                st.plotly_chart(fig_sens)
            
            st.success("Optimization complete!")
            
        except Exception as e:
            st.error(f"Optimization error: {str(e)}")
    
    # Final display of all figs if both
    if all_figures:
        st.subheader("All Visualizations")
        for fig in all_figures:
            if isinstance(fig, go.Figure):
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.pyplot(fig)

if __name__ == "__main__":
    main()
