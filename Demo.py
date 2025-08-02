import os
import warnings
import sys
import logging
import pandas as pd
import numpy as np
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
from urllib.parse import urlparse
import mlflow
import joblib
import dagshub
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, chi2_contingency

dagshub.init(repo_owner='Trilokuday3', repo_name='End_To_End_DATA_DRITF_Pipeline-dagshub', mlflow=True)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    rocauc = roc_auc_score(y_true, y_prob)
    mse = mean_squared_error(y_true, y_prob)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_prob)
    r2 = r2_score(y_true, y_prob)
    return acc, rocauc, mse, rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    data_path = r"C:\Users\trilo\Downloads\End To End DATA DRITF Pipeline\data\processed\Preprocessed_Data.csv"
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.exception("Unable to load processed data CSV, check your file path. Error: %s", e)
        sys.exit(1)

    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = df.drop(columns=numeric_features + ['Churn']).columns.tolist()

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Define parameter grids for each model
    param_grid = {
        "LogisticRegression": [
            {"max_iter": 1000, "C": 1.0},
            {"max_iter": 2000, "C": 0.5}
        ],
        "RandomForest": [
            {"n_estimators": 100, "max_depth": 10},
            {"n_estimators": 200, "max_depth": 20}
        ],
        "XGBoost": [
            {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}
        ]
    }

    remote_server_uri = "https://dagshub.com/Trilokuday3/End_To_End_DATA_DRITF_Pipeline.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)

    experiment_name = "TelcoChurnParamGridSearch"
    mlflow.set_experiment(experiment_name)

    model_map = {
        "LogisticRegression": LogisticRegression,
        "RandomForest": RandomForestClassifier,
        "XGBoost": XGBClassifier
    }

    os.makedirs("artifacts/drift_plots", exist_ok=True)

    for model_name, param_list in param_grid.items():
        for params in param_list:
            print(f"\n=== Running {model_name} with params: {params} ===\n")

            if model_name == "XGBoost":
                params = params.copy()
                params["use_label_encoder"] = False
                params["eval_metric"] = "logloss"
                params["random_state"] = 42
            else:
                params = params.copy()
                params["random_state"] = 42

            classifier = model_map[model_name](**params)
            model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
            with mlflow.start_run(run_name=f"{model_name}_{params}"):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                acc, rocauc, mse, rmse, mae, r2 = eval_metrics(y_test, y_pred, y_prob)

                metrics_dict = {
                    "accuracy": acc,
                    "roc_auc": rocauc,
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                }
                print(f"{model_name} ({params}) model results:")
                for k, v in metrics_dict.items():
                    print(f"{k}: {v:.4f}")
                print("Classification Report:\n", classification_report(y_test, y_pred))
                print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

                mlflow.log_param("model_type", model_name)
                mlflow.log_params(params)
                mlflow.log_metrics(metrics_dict)

                # Save and log the model as artifact
                model_path = os.path.join("artifacts", f"{model_name}_{str(params).replace(' ', '').replace(':', '').replace(',', '_')}_model.pkl")
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path, artifact_path="artifacts")
                os.remove(model_path)

                # ---------- START DATA DRIFT BLOCK ----------
                drift_results = []
                drift_plots_dir = "artifacts/drift_plots"

                # Numeric feature drift: KS test and density plots
                for col in numeric_features:
                    stat, p_value = ks_2samp(X_train[col], X_test[col])
                    drift_results.append({
                        'feature': col,
                        'type': 'numeric',
                        'statistic': stat,
                        'p_value': p_value
                    })
                    # Plot and log the drift plot
                    plt.figure()
                    X_train[col].plot(kind='density', label='Train', legend=True)
                    X_test[col].plot(kind='density', label='Test', legend=True)
                    plt.title(f'Drift plot: {col}')
                    plt.xlabel(col)
                    plt.legend()
                    drift_plot_path = os.path.join(drift_plots_dir, f"{model_name}_{col}_drift.png")
                    plt.savefig(drift_plot_path)
                    plt.close()
                    mlflow.log_artifact(drift_plot_path, artifact_path="artifacts/drift_plots")

                # Categorical feature drift: Chi-square test
                for col in categorical_features:
                    train_counts = X_train[col].value_counts()
                    test_counts = X_test[col].value_counts()
                    categories = list(set(train_counts.index).union(set(test_counts.index)))
                    train_freqs = train_counts.reindex(categories, fill_value=0)
                    test_freqs = test_counts.reindex(categories, fill_value=0)
                    table = np.array([train_freqs, test_freqs])
                    chi2, p_value, _, _ = chi2_contingency(table)
                    drift_results.append({
                        'feature': col,
                        'type': 'categorical',
                        'statistic': chi2,
                        'p_value': p_value
                    })

                # Save drift report csv
                drift_report_df = pd.DataFrame(drift_results)
                drift_csv = os.path.join("artifacts", f"drift_report_{model_name}_{str(params).replace(' ', '').replace(':', '').replace(',', '_')}.csv")
                drift_report_df.to_csv(drift_csv, index=False)
                mlflow.log_artifact(drift_csv, artifact_path="artifacts")
                # ---------- END DATA DRIFT BLOCK ----------
                
                
                
                # test1
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
# Corrected imports for Evidently 0.4.x
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

st.title("Telco Churn ML & Drift Interactive Dashboard (Evidently 0.4.x)")

# --- Data Loading and Preparation ---
# NOTE: You must update this path to where your actual data is stored.
# Using a placeholder for demonstration.
@st.cache_data
def load_data():
    try:
        # The user's original path. This will likely fail if not run on the original machine.
        data_path = r"C:\Users\trilo\Downloads\End To End DATA DRITF Pipeline\data\processed\Preprocessed_Data.csv"
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Could not find the data file at the specified path. Please update the 'data_path' variable in the script.")
        # As a fallback, try to load from a well-known source for demonstration purposes.
        st.warning("Loading a sample Telco Churn dataset from an online source for demonstration.")
        url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
        df = pd.read_csv(url)
        # Basic preprocessing to match the likely structure
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(subset=['TotalCharges'], inplace=True)
        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        df.drop(columns=['customerID'], inplace=True)
    return df

df = load_data()

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
# Dynamically find categorical features, excluding target and numeric
categorical_features = [col for col in df.columns if col not in numeric_features + ['Churn']]

X = df.drop(columns=['Churn'])
y = df['Churn']

# Ensure all specified numeric features are actually in the DataFrame
numeric_features = [feat for feat in numeric_features if feat in X.columns]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Model and Hyperparameter Selection ---
param_grid = {
    "LogisticRegression": [
        {"max_iter": 1000, "C": 1.0},
        {"max_iter": 2000, "C": 0.5}
    ],
    "RandomForest": [
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 20}
    ],
    "XGBoost": [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}
    ]
}
model_map = {
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
    "XGBoost": XGBClassifier
}

all_choices = []
for model_name, param_list in param_grid.items():
    for params in param_list:
        label = f"{model_name}: {params}"
        all_choices.append((model_name, params, label))

st.sidebar.header("Select Model and Parameters")
choice_idx = st.sidebar.selectbox(
    "Choose a model configuration:",
    range(len(all_choices)),
    format_func=lambda idx: all_choices[idx][2]
)
model_name, params, label = all_choices[choice_idx]

st.header(f"Results for: {label}")

# --- Preprocessing & Model Training Pipeline ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Set model-specific parameters
model_params = params.copy()
if model_name == "XGBoost":
    model_params["use_label_encoder"] = False
    model_params["eval_metric"] = "logloss"
    model_params["random_state"] = 42
else:
    model_params["random_state"] = 42
    
classifier = model_map[model_name](**model_params)
model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --- Model Evaluation Metrics ---
acc, rocauc, mse, rmse, mae, r2 = eval_metrics(y_test, y_pred, y_prob)
metrics_dict = {
    "Accuracy": acc,
    "ROC AUC Score": rocauc,
    "Mean Squared Error (MSE)": mse,
    "Root Mean Squared Error (RMSE)": rmse,
    "Mean Absolute Error (MAE)": mae,
    "R^2 Score": r2
}

st.subheader("Evaluation Metrics")
for k, v in metrics_dict.items():
    st.write(f"**{k}:** {v:.4f}")

st.subheader("Classification Report")
report_text = classification_report(y_test, y_pred, output_dict=False)
st.text(report_text)

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.colorbar(im)

# Add text annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
st.pyplot(fig)

# --- FINAL CORRECTED: Data Drift Table with Evidently 0.4.x ---
st.subheader("Data Drift Analysis")
st.write("""
This table shows the result of statistical tests comparing the distribution of each feature 
in the training data (reference) versus the test data (current).
""")

# 1. Create a Report object and add the DataDriftPreset
# This part is correct and does not need to change
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

# 2. Run the calculation
data_drift_report.run(reference_data=X_train, current_data=X_test)

# 3. Extract the results as a dictionary
drift_data = data_drift_report.as_dict()

# 4. FINAL CORRECTED LOGIC: Parse the dictionary based on the user's JSON output
st.write("Drift Score Table (Wasserstein for numerical, Jensen-Shannon for categorical)")

try:
    # THE FIX IS HERE: This path correctly points to the detailed results
    # in the second metric of the report.
    drift_details = drift_data['metrics'][1]['result']['drift_by_columns']

    drift_list = []
    # Loop through each feature's drift results
    for feature_name, metrics in drift_details.items():
        # Use .get() to safely access keys, providing 'N/A' if a key (like p_value) is missing.
        drift_list.append({
            'Feature': feature_name,
            'Feature Type': metrics.get('column_type', 'N/A'),
            'Statistic Used': metrics.get('stattest_name', 'N/A'),
            'Drift Score': f"{metrics.get('drift_score', 0):.6f}",
            'p-value': metrics.get('p_value', 'N/A'), # Will show 'N/A' as expected
            'Drift Detected': '✅ Yes' if metrics.get('drift_detected') else '❌ No'
        })
        
    drift_table = pd.DataFrame(drift_list)
    st.dataframe(drift_table)

except (KeyError, IndexError) as e:
    # This block will catch the error if the structure is unexpectedly different
    st.error(f"Caught an error while parsing the report: {e}")
    st.info("The structure of the Evidently report dictionary seems different than expected. This can happen between library versions.")
    st.write("Printing the raw dictionary structure below for debugging:")
    st.json(drift_data) # This will print the whole dictionary to help debug
    
except Exception as e:
    st.error(f"An unexpected error occurred while generating the drift report: {e}")


st.caption("This dashboard uses Evidently 0.4.12. Full HTML reports can also be exported if needed.")                