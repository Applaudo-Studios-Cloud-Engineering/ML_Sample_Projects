import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import create_preprocessing_pipeline_from_dict, create_feature_engineering_pipeline

# Load the model
os.chdir('../model')
model_name = os.listdir('.')[0]
model: DecisionTreeClassifier = pickle.load(open(model_name, 'rb'))

# Initialize the app
app = FastAPI()

# Declare the payload class
class Payload(BaseModel):
    PassengerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: str|None
    Embarked: str

@app.post('/')
def predict(payload: Payload):
    # Extract the payload
    X = payload.dict()
    for key in X:
        X[key] = [X[key]]
    # Create the preprocessing pipeline
    df = create_preprocessing_pipeline_from_dict(X, True)
    # Create the feature engineering pipeline
    df = create_feature_engineering_pipeline(df)
    # Predict
    y_pred = model.predict(df)
    return int(y_pred.argmax())