# End-to-End ML Pipeline (Airflow + MLflow + PostgreSQL)

An end-to-end machine learning pipeline that automates data ingestion, model training, hyperparameter tuning, experiment tracking, and model selection.

## 🚀 What it does

* Loads the Boston Housing dataset
* Stores data in PostgreSQL with deduplication
* Trains Ridge and XGBoost models with hyperparameter tuning
* Tracks experiments using MLflow
* Selects the best model based on RMSE
* Stores model performance metrics in the database

## 🛠 Tech stack

* Apache Airflow
* MLflow
* PostgreSQL
* scikit-learn
* XGBoost
* Pandas / NumPy

## 📁 Project structure

* `dags/` – Airflow DAG definition
* `requirements.txt` – Python dependencies
* `README.md` – Project documentation

## ⚙️ How to run

1. Start required services:

   * PostgreSQL
   * MLflow server
   * Airflow

2. Add the DAG file to your Airflow `dags/` directory

3. Configure Airflow connection:

   * Create a Postgres connection named `data-postgres`

4. Trigger the DAG:

   * `training_pipeline`

## 📊 What this demonstrates

* Building end-to-end ML pipelines
* Hyperparameter tuning with RandomizedSearchCV
* Experiment tracking with MLflow
* Data orchestration using Airflow
* Integration with PostgreSQL

## 📝 Notes

* The dataset is recreated on each run for simplicity
* Hyperparameter tuning is performed during each pipeline execution
* Designed for demonstration and learning purposes

