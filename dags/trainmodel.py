import numpy as np
import pandas as pd
import pendulum
from datetime import datetime
from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.postgresql import insert
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
from airflow.sdk import DAG, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from scipy.stats import randint, uniform
import mlflow.xgboost
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment('experiment')



@DAG(
    dag_id='training_pipeline',
    catchup=False,
    start_date=pendulum.datetime(2024, 5, 1, tz="UTC"),
    tags=['ml'],
    schedule='@daily'
)

def pipeline():

    @task
    def create_tables():
        hook = PostgresHook(postgres_conn_id='data-postgres')
        engine = hook.get_sqlalchemy_engine()

        df = pd.read_csv(url)
        df["row_hash"] = pd.util.hash_pandas_object(df, index=False).astype(str)
        df.head(0).to_sql("housing_data", engine, if_exists="replace", index=False)

        with engine.begin() as conn:
            conn.exec_driver_sql("""
            ALTER TABLE housing_data
            ADD CONSTRAINT unique_row_hash UNIQUE (row_hash)
            """)

            conn.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id SERIAL PRIMARY KEY,
                model_name TEXT,
                rmse FLOAT,
                r2 FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

    @task
    def load_data():
        hook = PostgresHook(postgres_conn_id='data-postgres')
        engine = hook.get_sqlalchemy_engine()

        df = pd.read_csv(url)
        df = df.drop_duplicates()

        num_cols = df.select_dtypes(include=['number']).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())

        df["row_hash"] = pd.util.hash_pandas_object(df, index=False).astype(str)

        metadata = MetaData()
        table = Table("housing_data", metadata, autoload_with=engine)

        stmt = insert(table).values(df.to_dict(orient="records"))
        stmt = stmt.on_conflict_do_nothing(constraint="unique_row_hash")
        
        with engine.begin() as conn:
            conn.execute(stmt)

    @task
    def train_model(_):

        with mlflow.start_run(run_name="model_comparison"):
            hook = PostgresHook(postgres_conn_id='data-postgres')
            engine = hook.get_sqlalchemy_engine()

            df = pd.read_sql('SELECT * FROM housing_data', engine)
            y = df['medv']
            X = df.drop(columns=['medv', 'row_hash'])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            with mlflow.start_run(run_name="ridge", nested=True):
                ridge_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge())
                ])
                param_distributions = {
                    "ridge__alpha": loguniform(1e-4, 1e3),
                    "ridge__fit_intercept": [True, False],
                    "ridge__solver": ["auto", "svd", "cholesky", "lsqr", "sag"]
                }

                searchridge = RandomizedSearchCV(
                    estimator=ridge_pipe,
                    param_distributions=param_distributions,
                    n_iter=30,
                    scoring="neg_mean_squared_error",   
                    cv=5,
                    random_state=42,
                    n_jobs=-1,
                    refit=True
                )

                searchridge.fit(X_train, y_train)
                ridge_preds = searchridge.predict(X_test)

                mlflow.log_params(searchridge.best_params_)

                ridge_rmse = float(np.sqrt(mean_squared_error(y_test, ridge_preds)))
                ridge_r2 = float(r2_score(y_test, ridge_preds))
                mlflow.log_metric('Ridge_RMSE', ridge_rmse)
                mlflow.log_metric('Ridge_R2', ridge_r2)
                mlflow.sklearn.log_model(searchridge, "model_ridge", registered_model_name='ridge_regression_versions')


            with mlflow.start_run(run_name="xgboost", nested=True):
                xgb = XGBRegressor(objective="reg:squarederror",random_state=42,n_jobs=1)
                param_distributions = {
                    "n_estimators": randint(100, 800),
                    "max_depth": randint(3, 10),
                    "learning_rate": uniform(0.01, 0.29),   
                    "subsample": uniform(0.6, 0.4),        
                    "colsample_bytree": uniform(0.6, 0.4),  
                    "min_child_weight": randint(1, 10),
                    "gamma": uniform(0, 5),
                    "reg_alpha": uniform(0, 2),
                    "reg_lambda": uniform(0.5, 3)
                }

                searchxgb = RandomizedSearchCV(
                    estimator=xgb,
                    param_distributions=param_distributions,
                    n_iter=40,
                    scoring="neg_mean_squared_error",
                    cv=5,
                    verbose=1,
                    random_state=42,
                    n_jobs=-1,
                    refit=True
                )

                searchxgb.fit(X_train, y_train)
                xgb_preds = searchxgb.predict(X_test)
                mlflow.log_params(searchxgb.best_params_)

                xgb_rmse = float(np.sqrt(mean_squared_error(y_test, xgb_preds)))
                xgb_r2 = float(r2_score(y_test, xgb_preds))
                mlflow.log_metric('Xgboost_RMSE', xgb_rmse)
                mlflow.log_metric('Xgboost_R2', xgb_r2)
                mlflow.xgboost.log_model(searchxgb.best_estimator_, "model_xgboost", registered_model_name='XGBregressor_versions')

            if ridge_rmse < xgb_rmse:
                best_model_name = "ridge"
                best_rmse = ridge_rmse
                best_r2 = ridge_r2
            else:
                best_model_name = "xgboost"
                best_rmse = xgb_rmse
                best_r2 = xgb_r2

            mlflow.log_param("best_model_name", best_model_name)
            mlflow.log_metric("best_model_rmse", best_rmse)
            mlflow.log_metric("best_model_r2", best_r2)

            return {
                "ridge": {"rmse": ridge_rmse, "r2": ridge_r2},
                "xgboost": {"rmse": xgb_rmse, "r2": xgb_r2},
                "best_model": {
                    "model_name": best_model_name,
                    "rmse": best_rmse,
                    "r2": best_r2
                }
            }

    @task
    def save_metrics(metrics):
        hook = PostgresHook(postgres_conn_id='data-postgres')
        engine = hook.get_sqlalchemy_engine()

        metadata = MetaData()
        table = Table("model_metrics", metadata, autoload_with=engine)

        rows = [
            {
                "model_name": "ridge",
                "rmse": metrics["ridge"]["rmse"],
                "r2": metrics["ridge"]["r2"],
                "created_at": datetime.utcnow()
            },
            {
                "model_name": "xgboost",
                "rmse": metrics["xgboost"]["rmse"],
                "r2": metrics["xgboost"]["r2"],
                "created_at": datetime.utcnow()
            },
            {
                "model_name": f'best:{metrics["best_model"]["model_name"]}',
                "rmse": metrics["best_model"]["rmse"],
                "r2": metrics["best_model"]["r2"],
                "created_at": datetime.utcnow()
            }
        ]

        stmt = insert(table).values(rows)

        with engine.begin() as conn:
            conn.execute(stmt)

        return 'metrics_saved'

    created = create_tables()
    loaded = load_data()
    created >> loaded

    metrics = train_model(loaded)
    save_metrics(metrics)

pipeline()