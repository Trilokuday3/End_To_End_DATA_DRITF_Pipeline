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
import mlflow
import joblib
import dagshub

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

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

    os.makedirs("artifacts", exist_ok=True)

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

                # ---------- START EVIDENTLY DATA DRIFT BLOCK ----------
                drift_report_html = os.path.join(
                    "artifacts", f"drift_report_{model_name}_{str(params).replace(' ', '').replace(':', '').replace(',', '_')}.html"
                )

                dashboard = Dashboard(tabs=[DataDriftTab()])
                dashboard.calculate(reference_data=X_train, current_data=X_test)
                dashboard.save(drift_report_html)

                mlflow.log_artifact(drift_report_html, artifact_path="artifacts")
                # ---------- END EVIDENTLY DATA DRIFT BLOCK ----------