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
    accuracy_score, confusion_matrix, roc_auc_score,
    classification_report, mean_squared_error, mean_absolute_error, r2_score
)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def eval_metrics(y_true, y_pred, y_prob):
    """Calculates and returns a variety of evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    rocauc = roc_auc_score(y_true, y_prob)
    mse = mean_squared_error(y_true, y_prob)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_prob)
    r2 = r2_score(y_true, y_prob)
    return acc, rocauc, mse, rmse, mae, r2

# The problematic line has been removed from here.
st.title("Telco Churn ML & Drift Interactive Dashboard (Evidently 0.4.x)")

# --- Data Loading and Preparation ---
@st.cache_data
def load_data():
    try:
        data_path = r"C:\Users\trilo\Downloads\End To End DATA DRITF Pipeline\data\processed\Preprocessed_Data.csv"
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Could not find the data file. Please update the 'data_path' variable.")
        st.warning("Loading a sample Telco Churn dataset from an online source for demonstration.")
        url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
        df = pd.read_csv(url)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(subset=['TotalCharges'], inplace=True)
        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        df.drop(columns=['customerID'], inplace=True)
    return df

df = load_data()

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in df.columns if col not in numeric_features + ['Churn']]

X = df.drop(columns=['Churn'])
y = df['Churn']

numeric_features = [feat for feat in numeric_features if feat in X.columns]
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

st.header(f"Results for: {label}")

# --- Preprocessing & Model Training Pipeline ---
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])

model_params = params.copy()
if model_name == "XGBoost":
    model_params.update({"use_label_encoder": False, "eval_metric": "logloss"})
model_params["random_state"] = 42
classifier = model_map[model_name](**model_params)
model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --- Model Evaluation Metrics ---
st.subheader("Evaluation Metrics")
acc, rocauc, mse, rmse, mae, r2 = eval_metrics(y_test, y_pred, y_prob)
st.write(f"**Accuracy:** {acc:.4f}")
st.write(f"**ROC AUC Score:** {rocauc:.4f}")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
# BEST PRACTICE: Create an explicit figure and pass it to st.pyplot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")
ax.set_title('Confusion Matrix'); ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
# Add annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
st.pyplot(fig)


# --- Data Drift Analysis Section ---
st.subheader("Data Drift Analysis")
data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=X_train, current_data=X_test)
drift_data = data_drift_report.as_dict()

st.write("Drift Score Table")
try:
    drift_details = drift_data['metrics'][1]['result']['drift_by_columns']
    drift_list = []
    for feature_name, metrics in drift_details.items():
         drift_list.append({
            'Feature': feature_name,
            'Type': metrics.get('column_type'),
            'Test': metrics.get('stattest_name'),
            'Score': metrics.get('drift_score'),
            'Drift': 'Yes' if metrics.get('drift_detected') else 'No'
        })
    drift_table = pd.DataFrame(drift_list)
    st.dataframe(drift_table)

    # --- Feature Drift Visualization Section ---
    st.subheader("Feature Drift Visualization")
    feature_names = list(drift_details.keys())
    selected_feature = st.selectbox("Select a feature to visualize its distribution drift:", feature_names)

    if selected_feature:
        feature_metrics = drift_details[selected_feature]
        
        # BEST PRACTICE: Create a new figure and axes for the plot
        fig_drift, ax_drift = plt.subplots()
        
        # --- ROBUST PLOTTING LOGIC ---
        def plot_dist(ax, distribution_data, label, linestyle='-'):
            x_data = distribution_data['x']
            y_data = distribution_data['y']
            
            # THE FIX: Ensure x and y have the same length to prevent crashing
            min_len = min(len(x_data), len(y_data))
            ax.plot(x_data[:min_len], y_data[:min_len], label=label, linestyle=linestyle, alpha=0.7)

        # Plot both distributions on the same axes
        plot_dist(ax_drift, feature_metrics['reference']['small_distribution'], 'Reference (Train)', linestyle='-')
        plot_dist(ax_drift, feature_metrics['current']['small_distribution'], 'Current (Test)', linestyle='--')
        
        ax_drift.set_title(f"Distribution for '{selected_feature}'")
        ax_drift.set_xlabel("Value")
        ax_drift.set_ylabel("Density / Frequency")
        ax_drift.legend()
        
        # Rotate labels if they are strings (categorical)
        if feature_metrics['column_type'] == 'cat':
            plt.setp(ax_drift.get_xticklabels(), rotation=45, ha="right")
            
        fig_drift.tight_layout()
        st.pyplot(fig_drift)

except (KeyError, IndexError) as e:
    st.error(f"Caught an error while parsing the report: {e}")
    st.info("The structure of the Evidently report dictionary seems different than expected.")
    st.write("Printing the raw dictionary structure below for debugging:")
    st.json(drift_data)
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

st.caption("This dashboard uses Evidently 0.4.12.")