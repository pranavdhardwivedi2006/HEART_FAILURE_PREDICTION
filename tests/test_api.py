import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_predict_endpoint_low_risk():
    payload = {
        "Age": 40,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 140,
        "Cholesterol": 289,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 172,
        "ExerciseAngina": "N",
        "Oldpeak": 0.0,
        "ST_Slope": "Up",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "HeartDisease" in data
    assert "Risk_Probability" in data


def test_predict_endpoint_high_risk():
    payload = {
        "Age": 55,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 160,
        "Cholesterol": 286,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 108,
        "ExerciseAngina": "Y",
        "Oldpeak": 1.5,
        "ST_Slope": "Flat",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["HeartDisease"] in [0, 1]