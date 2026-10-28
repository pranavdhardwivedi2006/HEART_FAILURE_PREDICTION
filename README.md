# Heart Failure Prediction API

An end-to-end Machine Learning pipeline that predicts the risk of heart disease based on clinical features. This project includes model training, a FastAPI prediction service, Docker containerization, automated testing with Pytest, and a GitHub Actions CI/CD pipeline.

## Dataset

**Source:** [Heart Failure Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

## Tech Stack

* **Machine Learning:** Scikit-Learn, Pandas, Joblib
* **API Framework:** FastAPI, Uvicorn, Pydantic
* **Testing:** Pytest, Httpx
* **DevOps:** Docker, GitHub Actions

## Project Structure

```text
.
├── .github/workflows/ci.yml   # GitHub Actions CI pipeline
├── tests/
│   └── test_api.py            # Pytest automated tests
├── Dockerfile                 # Docker container configuration
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── model.joblib               # Trained Random Forest model
├── scaler.joblib              # Feature scaler
└── model_columns.joblib       # Saved column structure
```

## Local Setup & Execution

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can access the interactive Swagger UI documentation at `http://127.0.0.1:8000/docs`.

### 3. Run Automated Tests

```bash
pytest
```

## Docker Execution

### 1. Build the Docker Image

```bash
docker build -t heart-failure-api .
```

### 2. Run the Docker Container

```bash
docker run -d -p 8000:8000 --name heart-container heart-failure-api
```

Access the API via `http://localhost:8000/docs`.

## API Endpoint Usage

**POST** `/predict`

**Sample Request Body (JSON):**

```json
{
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
  "ST_Slope": "Flat"
}
```

**Sample Response:**

```json
{
  "HeartDisease": 1,
  "Risk_Probability": 0.85
}
```
