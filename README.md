# End-to-End ML Pipeline (Airflow + MLflow + PostgreSQL + FastAPI)

An end-to-end machine learning pipeline that automates data ingestion, model training, hyperparameter tuning, experiment tracking, model selection, and model serving.

## What it does

* Loads the Boston Housing dataset
* Stores data in PostgreSQL with deduplication
* Trains Ridge and XGBoost models with hyperparameter tuning
* Tracks experiments using MLflow
* Selects the best model based on RMSE
* Promotes the best model to `@champion` in MLflow Model Registry
* Stores model performance metrics in the database
* Serves predictions via a FastAPI REST API

## Tech stack

* Apache Airflow
* MLflow
* PostgreSQL
* FastAPI + Uvicorn
* scikit-learn
* XGBoost
* Pandas / NumPy
* Docker / Docker Compose

## Project structure

* `dags/` – Airflow DAG definition
* `serving/` – FastAPI model serving service
  * `serve.py` – FastAPI app
  * `Dockerfile` – Container for the API
* `requirements.txt` – Airflow dependencies
* `docker-compose.yaml` – All services defined
* `README.md` – Project documentation