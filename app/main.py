import os
from typing import List, Optional

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# import dotenv

# # Load environment variables from a .env file
# dotenv.load_dotenv()

# Harcode the prod model path and encoder path here because it's only used in this file so far
# Will consider moving the hardcoded paths to environment variables if neccessary
PROD_MODEL_PATH = './prod_models/lgb_model.joblib'
PROD_ENCODER_PATH = './prod_models/encoder.joblib'

# Load the saved model and encoder
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

app = FastAPI()

# Define the input schema with None as the default value
class LoanApplication(BaseModel):
    loanId: Optional[int] = None
    anon_ssn: Optional[str] = None
    payFrequency: Optional[str] = None
    apr: Optional[float] = None
    applicationDate: Optional[str] = None
    originatedDate: Optional[str] = None
    nPaidOff: Optional[int] = None
    loanStatus: Optional[str] = None
    loanAmount: Optional[float] = None
    originallyScheduledPaymentAmount: Optional[float] = None
    state: Optional[str] = None
    leadType: Optional[str] = None
    leadCost: Optional[float] = None
    fpStatus: Optional[str] = None
    clearfraudscore: Optional[float] = None


def preprocess_single_input(df: pd.DataFrame):
    """
    Preprocess a single input dataframe.

    Args:
    - df (pd.DataFrame): Input dataframe.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    """
    # Convert columns to datetime format
    df['originatedDate'] = pd.to_datetime(df['originatedDate'], format='mixed')
    df['applicationDate'] = pd.to_datetime(df['applicationDate'], format='mixed')
    
    # Calculate the difference in hours
    df['app_processing_hours'] = (df['originatedDate'] - df['applicationDate']).dt.total_seconds() / 3600
    
    return df

# List of all expected columns
expected_columns = ['apr', 'loanAmount', 'originallyScheduledPaymentAmount', 'leadCost', 'app_processing_hours', 'clearfraudscore', 'payFrequency', 'nPaidOff', 'state', 'fpStatus']

def prepare_input_data(application_dict):
    # Convert the input data to a DataFrame
    data = pd.DataFrame([application_dict])
    
    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in data.columns or data[col].isnull().all():
            data[col] = np.nan
    
    return data


@app.post("/predict")
def predict(application: LoanApplication):
    try:
        # Prepare the input data
        data = prepare_input_data(application.dict())

        # Preprocess the single input
        data = preprocess_single_input(data)

        # Encode the data using the loaded encoder
        encoded_data = encoder.transform(data)

        # Make predictions using the loaded model
        predictions = model.predict(encoded_data)

        return {"prediction": predictions[0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

