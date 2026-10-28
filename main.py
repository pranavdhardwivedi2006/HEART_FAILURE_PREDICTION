import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Heart Failure Prediction API")

# Load model, scaler, and expected column structure
model = joblib.load("models/model.joblib")
scaler = joblib.load("models/scaler.joblib")
model_columns = joblib.load("models/model_columns.joblib")


# Define incoming request body structure based on your raw dataset features
class PatientData(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str


@app.post("/predict")
def predict_heart_failure(data: PatientData):
    # 1. Convert input JSON to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # 2. Apply One-Hot Encoding (same as training)
    input_encoded = pd.get_dummies(input_df)

    # 3. Reindex to match exact columns the model was trained on (fills missing dummy columns with 0)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # 4. Scale the features
    input_scaled = scaler.transform(input_encoded)

    # 5. Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    return {
        "HeartDisease": int(prediction[0]),
        "Risk_Probability": float(probability[0][1]),
    }