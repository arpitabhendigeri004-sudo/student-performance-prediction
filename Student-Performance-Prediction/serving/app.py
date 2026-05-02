from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize app
app = FastAPI()

# ✅ CORS FIX (MUST BE HERE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("models/student_model.pkl")

# Input schema
class StudentInput(BaseModel):
    age: int
    studytime: int
    failures: int
    absences: int
    G1: int
    G2: int
    school: str
    sex: str
    address: str
    famsize: str
    Pstatus: str

# Home route
@app.get("/")
def home():
    return {"message": "Student Performance API Running"}

# Prediction route
@app.post("/predict")
def predict(data: StudentInput):
    try:
        input_df = pd.DataFrame([data.model_dump()])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "status": "Pass" if prediction == 1 else "Fail"
        }

    except Exception as e:
        return {"error": str(e)}