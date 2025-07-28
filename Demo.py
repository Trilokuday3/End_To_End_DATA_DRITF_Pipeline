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
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    classification_report, mean_squared_error, mean_absolute_error, r2_score
)
from urllib.parse import urlparse
import mlflow
import joblib
import dagshub

dagshub.init(repo_owner='Trilokuday3', repo_name='ML_Flow-dagshub', mlflow=True)

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
    # Read the processed telco churn dataset
    data_path = r"C:\Users\trilo\Downloads\End To End DATA DRITF Pipeline\data\processed\Preprocessed_Data.csv"
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.exception(
            "Unable to load processed data CSV, check your file path. Error: %s", e
        )
        sys.exit(1)

    print("Data shape after loading:", df.shape)
    print("First few rows:\n", df.head())
    print("Churn unique values:", df['Churn'].unique())
    print("NaNs per column:\n", df.isna().sum())

    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = df.drop(columns=numeric_features + ['Churn']).columns.tolist()

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    print("X shape:", X.shape)
    print("y shape:", y.shape)

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

    # Pick model
    model_name = sys.argv[1] if len(sys.argv) > 1 else "LogisticRegression"
    if model_name.lower() == "randomforest":
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        classifier = LogisticRegression(max_iter=1000, random_state=42)
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

    remote_server_uri = "https://dagshub.com/Trilokuday3/ML_Flow-dagshub.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)

    with mlflow.start_run():
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
        print(f"{model_name} model results:")
        for k, v in metrics_dict.items():
            print(f"{k}: {v:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        mlflow.log_param("model_type", model_name)
        mlflow.log_params(classifier.get_params())
        mlflow.log_metrics(metrics_dict)

        # Save model with joblib and log as artifact
        os.makedirs("model_dir", exist_ok=True)
        model_path = os.path.join("model_dir", "model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")
        os.remove(model_path)