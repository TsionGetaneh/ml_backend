from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()

# CORS middleware for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models from model/ folder
lr_model = joblib.load("model/logistic_model.pkl")
dt_model = joblib.load("model/decision_tree_model.pkl")

@app.get("/")
def home():
    return {"message": "ML Model API is running"}

@app.post("/predict/logistic")
def predict_logistic(features: list):
    data = np.array(features).reshape(1, -1)
    prediction = lr_model.predict(data)
    return {"prediction": int(prediction[0])}

@app.post("/predict/dt")
def predict_dt(features: list):
    data = np.array(features).reshape(1, -1)
    prediction = dt_model.predict(data)
    return {"prediction": int(prediction[0])}
