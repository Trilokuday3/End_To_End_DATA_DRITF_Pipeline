import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score
)

# Evidently Imports
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
import streamlit.components.v1 as components

# Import individual metrics for detailed error analysis
from evidently.metrics import (
    RegressionPredictedVsActualPlot,
    RegressionErrorDistribution,
)
import traceback

# Helper Function to Generate Reports
def generate_evidently_report_html(metrics, current_data, reference_data, column_mapping):
    """Generates and returns the HTML for an Evidently report."""
    report = Report(metrics=metrics)
    report.run(current_data=current_data, reference_data=reference_data, column_mapping=column_mapping)
    return report.get_html()

st.set_page_config(layout="wide")
st.title("Telco Churn: ML Performance & Error Analysis Dashboard")

# --- Data Loading and Preparation (This is correctly cached) ---
@st.cache_data
def load_data():
    try:
        data_path = "Preprocessed_Data.csv"
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.warning("Local data file not found. Loading a sample Telco Churn dataset from an online source.")
        url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
        df = pd.read_csv(url)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(subset=['TotalCharges'], inplace=True)
        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        df.drop(columns=['customerID'], inplace=True)
    return df

df_loaded = load_data()
df = df_loaded.copy()

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in df.columns if col not in numeric_features + ['Churn']]

# Explicitly set data types to prevent ambiguity
for col in categorical_features:
    df[col] = df[col].astype(str)

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Model and Hyperparameter Selection ---
param_grid = {
    "LogisticRegression": [{"max_iter": 1000, "C": 1.0}, {"max_iter": 2000, "C": 0.5}],
    "RandomForest": [{"n_estimators": 100, "max_depth": 10}, {"n_estimators": 200, "max_depth": 20}],
    "XGBoost": [{"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}, {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}]
}
model_map = {"LogisticRegression": LogisticRegression, "RandomForest": RandomForestClassifier, "XGBoost": XGBClassifier}

all_choices = [(model_name, params, f"{model_name}: {params}") for model_name, param_list in param_grid.items() for params in param_list]

st.sidebar.header("Select Model and Parameters")
choice_idx = st.sidebar.selectbox("Choose a model configuration:", range(len(all_choices)), format_func=lambda idx: all_choices[idx][2])
model_name, params, label = all_choices[choice_idx]

# --- Preprocessing & Model Training ---
with st.spinner(f"Training {model_name}..."):
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='passthrough')
    
    model_params = params.copy()
    if model_name == "XGBoost": model_params.update({"use_label_encoder": False, "eval_metric": "logloss"})
    model_params["random_state"] = 42
    classifier = model_map[model_name](**model_params)
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    
    model.fit(X_train, y_train)

# --- Prepare DataFrames with Predictions for all Reports ---
y_train_pred = model.predict(X_train)
y_train_pred_proba = model.predict_proba(X_train)[:, 1]
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

current_data = X_test.copy(); current_data['Churn'] = y_test; current_data['prediction'] = y_test_pred; current_data['prediction_proba'] = y_test_pred_proba
reference_data = X_train.copy(); reference_data['Churn'] = y_train; reference_data['prediction'] = y_train_pred; reference_data['prediction_proba'] = y_train_pred_proba

# --- REPORT SECTION 1: Model Performance Report ---
st.header("Evidently Model Performance Report")
with st.spinner("Generating Performance Report..."):
    try:
        classification_mapping = ColumnMapping()
        classification_mapping.target = 'Churn'
        classification_mapping.prediction = 'prediction'
        classification_mapping.prediction_probas = 'prediction_proba'
        classification_mapping.numerical_features = numeric_features
        classification_mapping.categorical_features = categorical_features

        report_html = generate_evidently_report_html(
            metrics=[ClassificationPreset()],
            current_data=current_data,
            reference_data=reference_data,
            column_mapping=classification_mapping
        )
        components.html(report_html, height=800, scrolling=True)
    except Exception as e:
        st.error(f"Could not generate the Model Performance Report: {e}")

# --- REPORT SECTION 2: Detailed Model Error Analysis ---
st.header("Detailed Model Error Analysis")
st.write("This section treats the model's **predicted probability** as a continuous value to analyze calibration and error behavior.")

regression_mapping = ColumnMapping()
regression_mapping.target = 'Churn'
regression_mapping.prediction = 'prediction_proba'
regression_mapping.numerical_features = numeric_features
regression_mapping.categorical_features = categorical_features

col1, col2 = st.columns(2)

with col1:
    with st.spinner("Generating Predicted vs. Actual plot..."):
        try:
            st.subheader("Predicted vs. Actual Probability")
            report_html = generate_evidently_report_html(
                metrics=[RegressionPredictedVsActualPlot()],
                current_data=current_data, reference_data=None, column_mapping=regression_mapping
            )
            components.html(report_html, height=450, scrolling=True)
        except Exception as e:
            st.error(f"Plot failed: {e}")

with col2:
    with st.spinner("Generating Error Distribution plot..."):
        try:
            st.subheader("Error Distribution")
            report_html = generate_evidently_report_html(
                metrics=[RegressionErrorDistribution()],
                current_data=current_data, reference_data=None, column_mapping=regression_mapping
            )
            components.html(report_html, height=450, scrolling=True)
        except Exception as e:
            st.error(f"Plot failed: {e}")

# --- REPORT SECTION 3: Manual Error Bias Analysis ---
st.header("Manual Error Bias Analysis")
st.write("This section manually calculates model performance for different data segments to robustly identify bias.")

def display_manual_bias_analysis(feature_name, dataframe):
    """Calculates and displays a manual error bias table using Pandas."""
    if not feature_name:
        return

    st.markdown(f"#### Breakdown by **{feature_name}** on the Test Set")

    if feature_name in numeric_features:
        try:
            dataframe[f'{feature_name}_bins'] = pd.cut(dataframe[feature_name], bins=5)
            analysis_feature = f'{feature_name}_bins'
        except Exception as e:
            st.warning(f"Could not bin numerical feature '{feature_name}'. Error: {e}")
            return
    else:
        analysis_feature = feature_name
        
    dataframe['error'] = dataframe['prediction_proba'] - dataframe['Churn']
    
    bias_report = dataframe.groupby(analysis_feature).agg(
        Sample_Count=('Churn', 'count'),
        Actual_Churn_Rate=('Churn', 'mean'),
        Avg_Predicted_Prob=('prediction_proba', 'mean'),
        Mean_Error_Bias=('error', 'mean'),
    ).reset_index()

    bias_report['Actual_Churn_Rate'] = bias_report['Actual_Churn_Rate'].map('{:.2%}'.format)
    bias_report['Avg_Predicted_Prob'] = bias_report['Avg_Predicted_Prob'].map('{:.2%}'.format)
    bias_report['Mean_Error_Bias'] = bias_report['Mean_Error_Bias'].map('{:+.4f}'.format)

    st.dataframe(bias_report)

col_num, col_cat = st.columns(2)

with col_num:
    selected_num_feature = st.selectbox("Select a numerical feature:", [None] + numeric_features, key="num_bias")
    display_manual_bias_analysis(selected_num_feature, current_data.copy())

with col_cat:
    selected_cat_feature = st.selectbox("Select a categorical feature:", [None] + categorical_features, key="cat_bias")
    display_manual_bias_analysis(selected_cat_feature, current_data.copy())

st.caption("Error Bias Analysis is calculated manually with Pandas for robustness.")