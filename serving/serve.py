from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

model = None

def get_model():
    global model
    if model is None:
        try:
            model = mlflow.pyfunc.load_model("models:/housing_price_model@champion")
        except Exception:
            raise HTTPException(status_code=503, detail="Model not ready yet, DAG still running")
    return model

class HouseFeatures(BaseModel):
    crim: float
    zn: float
    indus: float
    chas: float
    nox: float
    rm: float
    age: float
    dis: float
    rad: float
    tax: float
    ptratio: float
    b: float
    lstat: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: HouseFeatures):
    df = pd.DataFrame([features.model_dump()])
    prediction = get_model().predict(df)
    return {"predicted_price": round(float(prediction[0]), 4)}