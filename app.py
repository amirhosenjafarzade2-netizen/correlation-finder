# app.py
"""
Main Streamlit app: UI, orchestration, display.
Integrates all modules; handles inputs, runs tasks, shows results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import io  # For download buffers
from typing import List
from config import Config, DEFAULT_CONFIG, PRESSURE_COLS
from data import load_and_preprocess_data, detect_pressure_column, classify_regimes
from analysis import analyze_relationships
from models import train_neural_network, compute_shap_values, predict_nn
from optimization import optimize_target
from visualization import generate_viz_from_analysis
from utils import logger, MissingDataError, InvalidPressureError, impute_missing_values
from formula_discovery import discover_formula, PYSR_AVAILABLE, GPLEARN_AVAILABLE  # Added imports for availability

# Configure Plotly for Streamlit
pio.renderers.default = 'browser'  # Or 'notebook' if needed

st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

def _build_config_from_sidebar() -> Config:
    """Build Config from sidebar inputs (no caching, as it contains widgets)."""
    st.sidebar.header("Configuration")
    random_seed = st.sidebar.number_input("Random Seed", value=DEFAULT_CONFIG["random_seed"])
    min_rows = st.sidebar.number_input("Min Rows", value=DEFAULT_CONFIG["min_rows"], min_value=1)
    epochs = st.sidebar.number_input("NN Epochs", value=DEFAULT_CONFIG["epochs"], min_value=1)
    validation_split = st.sidebar.slider("Validation Split", 0.0, 1.0, DEFAULT_CONFIG["validation_split"])
    pop_size = st.sidebar.number_input("GA Pop Size", value=DEFAULT_CONFIG["pop_size"], min_value=1)
    n_generations = st.sidebar.number_input("GA Generations", value=DEFAULT_CONFIG["n_generations"], min_value=1)
    n_calls = st.sidebar.number_input("Bayesian Calls", value=DEFAULT_CONFIG["n_calls"], min_value=1)
    nn_layers_input = st.sidebar.text_input("NN Layers (comma-separated)", value=",".join(map(str, DEFAULT_CONFIG["nn_layers"])))
    nn_layers = [int(x.strip()) for x in nn_layers_input.split(",")]
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
    
    config = _build_config_from_sidebar()  # No caching; widgets inside
    rng = np.random.default_rng(config.random_seed)
    
    # File upload
    uploaded_files = st.file_uploader("Upload Excel files", accept_multiple_files=True, type=['xlsx', 'xls'])
    
    # Row sampling
    n_rows_input = st.number_input("Number of rows to sample (0 for all)", min_value=0, value=0)
    n_rows = None if n_rows_input == 0 else n_rows_input
    
    if st.button("Load Data"):
        try:
            df, params = load_and_preprocess_data(uploaded_files, n_rows, config, rng)
            st.session_state.df = df
            st.session_state.params = params
            st.success(f"Loaded {len(df)} rows with {len(params)} parameters: {', '.join(params)}")
            
            # Data preview
            with st.expander("Data Preview"):
                st.dataframe(df.head())
                # Simple missing % example
                missing_df = pd.DataFrame({
                    'Column': params,
                    'Missing %': [df[col].isna().sum() / len(df) * 100 for col in params]
                })
                st.dataframe(missing_df)
            
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
            logger.error("Data load failed", error=str(e))
            return
    
    if 'df' not in st.session_state:
        st.warning("Please load data first.")
        return
    
    df = st.session_state.df
    params = st.session_state.params
    pressure_col = st.session_state.get('pressure_col')
    bubble_point = st.session_state.get('bubble_point', config.bubble_point_default)
    
    # Task selection
    task_options = ["Select a Task"] + ["1: Relationship Analysis", "2: Target Optimization", 
                                        "3: Symbolic Formula Discovery", "4: Both"]
    task = st.selectbox("Select Task", task_options, index=0)  # Default to "Select a Task"
    
    run_analysis = task in ["1: Relationship Analysis", "4: Both"]
    run_opt = task in ["2: Target Optimization", "4: Both"]
    run_formula = task in ["3: Symbolic Formula Discovery", "4: Both"]
    
    if st.button("Run Selected Task"):
        all_figures = []
        save_files = st.checkbox("Save Results to Files", value=False)
        interactive = st.checkbox("Enable Interactive Exploration", value=False)
        ml_method = st.selectbox("ML Method for Surrogates/Importance", ["nn", "rf", "lr", "svm"], index=1)  # Default to rf
        optimizer = st.selectbox("Optimizer", ["ga", "bayesian"], index=0)  # Default to ga
        target_name = st.selectbox("Target Parameter", params, index=0)
        
        if run_analysis:
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                status_text.text("Analyzing relationships...")
                results = analyze_relationships(df, params, pressure_col, bubble_point, config, ml_method=ml_method)
                progress_bar.progress(0.5)
                
                status_text.text("Generating visualizations...")
                figs = generate_viz_from_analysis(results, params, config)
                all_figures.extend(figs)
                progress_bar.progress(1.0)
                
                st.subheader("Analysis Results")
                # Display figures with unique keys to avoid duplicate ID
                for i, fig in enumerate(figs):
                    if isinstance(fig, go.Figure):
                        st.plotly_chart(fig, use_container_width=True, key=f"analysis_plotly_{i}")
                    else:
                        st.pyplot(fig, key=f"analysis_pyplot_{i}")
                
                # Downloads if save_files
                if save_files:
                    # Correlation matrix
                    csv_buffer = io.StringIO()
                    results['corr_matrix'].to_csv(csv_buffer, index=True)
                    st.download_button(
                        "Download Correlation Matrix",
                        csv_buffer.getvalue(),
                        "correlation_matrix.csv",
                        "text/csv",
                        key="download_corr"
                    )
                    # MI matrix
                    csv_buffer = io.StringIO()
                    results['mi_matrix'].to_csv(csv_buffer, index=True)
                    st.download_button(
                        "Download MI Matrix",
                        csv_buffer.getvalue(),
                        "mi_matrix.csv",
                        "text/csv",
                        key="download_mi"
                    )
                    # MI saturated
                    if not results['mi_saturated'].isna().all().all():
                        csv_buffer = io.StringIO()
                        results['mi_saturated'].to_csv(csv_buffer, index=True)
                        st.download_button(
                            "Download MI Saturated Matrix",
                            csv_buffer.getvalue(),
                            "mi_saturated_matrix.csv",
                            "text/csv",
                            key="download_mi_saturated"
                        )
                    # MI undersaturated
                    if not results['mi_undersaturated'].isna().all().all():
                        csv_buffer = io.StringIO()
                        results['mi_undersaturated'].to_csv(csv_buffer, index=True)
                        st.download_button(
                            "Download MI Undersaturated Matrix",
                            csv_buffer.getvalue(),
                            "mi_undersaturated_matrix.csv",
                            "text/csv",
                            key="download_mi_undersaturated"
                        )
                    # RF importance
                    if results['rf_importance']:
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            for target, importance_df in results['rf_importance'].items():
                                importance_df.to_excel(writer, sheet_name=target[:31], index=False)
                        st.download_button(
                            "Download Feature Importance",
                            excel_buffer.getvalue(),
                            "feature_importance.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_rf_importance"
                        )
                
                st.success("Analysis complete!")
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                logger.error("Analysis failed", error=str(e))
                progress_bar.empty()
                status_text.empty()
        
        if run_opt:
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                status_text.text("Training surrogate model...")
                optimal_df, opt_figs, shap_data = optimize_target(df, params, target_name, config, optimizer, ml_method=ml_method)
                progress_bar.progress(0.4)
                
                status_text.text("Running optimization...")
                progress_bar.progress(0.8)
                
                all_figures.extend(opt_figs)
                
                st.subheader("Optimal Parameters")
                st.dataframe(optimal_df)
                for i, fig in enumerate(opt_figs):
                    st.pyplot(fig, key=f"opt_pyplot_{i}")
                
                if save_files:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        optimal_df.to_excel(writer, index=False)
                    st.download_button(
                        "Download Optimal Params",
                        excel_buffer.getvalue(),
                        f"optimal_{target_name}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_optimal"
                    )
                
                if interactive:
                    st.subheader("Interactive Parameter Exploration")
                    features = [p for p in params if p != target_name]
                    model, scaler = train_neural_network(df[features], df[target_name], config)
                    test_values = {}
                    for feat in features:
                        test_values[feat] = st.slider(
                            feat, 
                            float(df[feat].min()), 
                            float(df[feat].max()), 
                            float(df[feat].mean()),
                            key=f"slider_{feat}"
                        )
                    
                    X_test = pd.DataFrame([test_values], columns=features)
                    X_test_scaled = scaler.transform(X_test)
                    pred = predict_nn(model, X_test_scaled, config.batch_size)[0][0]
                    st.metric(f"Predicted {target_name}", f"{pred:.4f}")
                    
                    # Sensitivity plot
                    selected_feat = st.selectbox("Sensitivity to", features, key="sensitivity_select")
                    feat_range = np.linspace(df[selected_feat].min(), df[selected_feat].max(), 50)
                    sens_preds = []
                    for val in feat_range:
                        temp_df = X_test.copy()
                        temp_df[selected_feat] = val
                        temp_scaled = scaler.transform(temp_df)
                        sens_preds.append(predict_nn(model, temp_scaled, config.batch_size)[0][0])
                    fig_sens = px.line(
                        x=feat_range, 
                        y=sens_preds, 
                        title=f"Sensitivity of {target_name} to {selected_feat}",
                        labels={'x': selected_feat, 'y': f'Predicted {target_name}'}
                    )
                    st.plotly_chart(fig_sens, use_container_width=True, key="sensitivity_plot")
                
                st.success("Optimization complete!")
                progress_bar.progress(1.0)
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"Optimization error: {str(e)}")
                logger.error("Optimization failed", error=str(e))
                progress_bar.empty()
                status_text.empty()
        
        if run_formula:
            try:
                st.subheader("Symbolic Formula Discovery")
                features = [p for p in params if p != target_name]
                if len(features) < 1:
                    st.warning("Need at least one feature for formula discovery.")
                else:
                    methods = ["linear"]  # Always available
                    if GPLEARN_AVAILABLE:
                        methods.append("gplearn")
                    if PYSR_AVAILABLE:
                        methods.append("pysr")
                    if not methods:
                        st.error("No formula discovery methods available.")
                    else:
                        discovery_method = st.selectbox("Discovery Method", methods, index=0, key="formula_method")
                        formula = discover_formula(df[features], df[target_name], features, method=discovery_method, target_name=target_name)
                        st.latex(formula['str_formula'])
                        st.metric("Fit Score (R²)", f"{formula['score']:.4f}")
            except Exception as e:
                st.error(f"Formula discovery error: {str(e)}")
                logger.error("Formula discovery failed", error=str(e), method=discovery_method)
    else:
        if task == "Select a Task":
            st.info("Please select a task and click 'Run Selected Task' to begin.")

if __name__ == "__main__":
    main()
