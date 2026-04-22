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

from airflow.sdk import DAG, task
from airflow.providers.postgres.hooks.postgres import PostgresHook


url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'

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
        hook = PostgresHook(postgres_conn_id='data-postgres')
        engine = hook.get_sqlalchemy_engine()

        df = pd.read_sql('SELECT * FROM housing_data', engine)
        y = df['medv']
        X = df.drop(columns=['medv', 'row_hash'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ridge = Ridge()
        ridge.fit(X_train, y_train)
        ridge_preds = ridge.predict(X_test)
        ridge_rmse = float(np.sqrt(mean_squared_error(y_test, ridge_preds)))
        ridge_r2 = float(r2_score(y_test, ridge_preds))

        xgb = XGBRegressor()
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        xgb_rmse = float(np.sqrt(mean_squared_error(y_test, xgb_preds)))
        xgb_r2 = float(r2_score(y_test, xgb_preds))

        return {
            "ridge": {"rmse": ridge_rmse, "r2": ridge_r2},
            "xgboost": {"rmse": xgb_rmse, "r2": xgb_r2}
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
            }
        ]

        stmt = insert(table).values(rows)

        with engine.begin() as conn:
            conn.execute(stmt)

    created = create_tables()
    loaded = load_data()
    created >> loaded

    metrics = train_model(loaded)
    save_metrics(metrics)

pipeline()