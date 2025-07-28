import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "drift_ml_project"

list_of_files = [
    "data/raw/.gitkeep",                             # Keep "data/raw" directory in git, even when empty
    "data/processed/.gitkeep",                       # Keep "data/processed" directory in git, even when empty
    "drift_monitoring/__init__.py",
    "drift_monitoring/drift_detector.py",            # Script for drift detection logic
    "drift_monitoring/report_gen.py",
    "model/__init__.py",
    "model/train.py",                                # Model training script
    "model/retrain.py",                              # Retraining logic if needed
    "model/evaluate.py",                             # Evaluation script
    "deployment/__init__.py",
    "deployment/api.py",                             # API (FastAPI/Flask) serving script
    "deployment/Dockerfile",
    "deployment/requirements.txt",
    "deployment/start.sh",
    "pipelines/__init__.py",
    "pipelines/orchestrate.py",                      # E.g. Airflow DAG or Prefect Flow
    "notebooks/EDA.ipynb",                           # Exploratory Data Analysis notebook
    "notebooks/Drift_Reports.ipynb",                 # Notebook for drift report viewing
    "README.md",
    "requirements.txt",
    "main.py",                                       # Entrypoint if needed
    ".gitignore"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directory if not exists
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # Create empty file if not exists or is zero-size
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")