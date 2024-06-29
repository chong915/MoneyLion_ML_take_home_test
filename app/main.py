import os
from typing import List, Optional

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # # Hardcoded seed data
# seed_data = {
#     'payFrequency_target': 0.459830,
#     'nPaidOff_target': 0.642234,
#     'state_target': 0.569760,
#     'fpStatus_target': 0.441129,
#     # 'leadType_target': 0.484141,
#     'payFrequency_freq': 2477.000000,
#     'nPaidOff_freq': 21.000000,
#     'state_freq': 5992.000000,
#     'fpStatus_freq': 29828.000000,
#     # 'leadType_freq': 8828.000000,
#     'apr': 478.670000,
#     'loanAmount': 500.000000,
#     'originallyScheduledPaymentAmount': 2249.940000,
#     'leadCost': 25.000000,
#     'app_processing_hours': 18.165733,
#     'clearfraudscore': 100
# }

# num_feats = ['apr', 'loanAmount', 'originallyScheduledPaymentAmount', 'leadCost', 'app_processing_hours', 'clearfraudscore']
# freq_feats = ['payFrequency', 'nPaidOff', 'state', 'fpStatus', 'leadType']

# # Hardcoded seed data
# seed_data = {
#     "payFrequency": "B",
#     "apr": 50.5,
#     "applicationDate": "2023-06-15",
#     "originatedDate": "2023-06-15",
#     "nPaidOff": 0,
#     "loanAmount": 10000.0,
#     "originallyScheduledPaymentAmount": 300.0,
#     "state": "FL",
#     "leadType": "lead",
#     "leadCost": 50.0,
#     "fpStatus": "No Payments",
#     "clearfraudscore": 1000.0
# }

# Harcode the prod model path and encoder path here because it's only used in this file so far
# Will consider moving the hardcoded paths to environment variables if neccessary
PROD_MODEL_PATH = './models/lgb_model.joblib'
PROD_ENCODER_PATH = './models/encoder.joblib'

# Load the saved model and encoder
model = joblib.load(PROD_MODEL_PATH)
encoder = joblib.load(PROD_ENCODER_PATH)

# Check the attributes of the loaded encoder
print("Numerical Features:", encoder.num_feats)
print("Target Encoded Features:", encoder.target_feats)
print("Frequency Encoded Features:", encoder.freq_feats)
print("Target Encoder:", encoder.target_encoder)
print("Frequency Encoder:", encoder.freq_encoder)

app = FastAPI()

logging.info(f"Current working directory : {os.getcwd()}")

# Define the input schema with None as the default value
class LoanApplication(BaseModel):
    """
    Schema for the loan application input data.
    """
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
expected_columns = ['apr', 'loanAmount', 'originallyScheduledPaymentAmount', 'leadCost', 'originatedDate', 'applicationDate', 'clearfraudscore', 'payFrequency', 'nPaidOff', 'state', 'fpStatus']

def prepare_input_data(application_dict) -> pd.DataFrame:
    """
    Prepare the input data for prediction.

    Args:
    - application_dict (dict): Input application data as a dictionary.

    Returns:
    - pd.DataFrame: Prepared input data as a DataFrame.
    """
    logging.info("Preparing input data for prediction.")

    # Convert the input data to a DataFrame
    data = pd.DataFrame([application_dict])
    
    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in data.columns or data[col].isnull().all():
            data[col] = np.nan
    
    logging.info("Input data preparation completed.")
    return data


@app.post("/predict")
def predict(application: LoanApplication) -> dict:
    """
    Predict the loan risk using the provided application data.

    Args:
    - application (LoanApplication): The loan application data.

    Returns:
    - dict: The prediction result.
    """
    try:
        logging.info("Received prediction request.")
        
        # Prepare the input data
        data = prepare_input_data(application.dict())

        # encoded_data = pd.DataFrame([seed_data])

        # Encode the data using the loaded encoder
        encoded_data = encoder.transform(data)

        logging.info(f"Encoded data:\n{encoded_data.to_string(index=False)}")

        # Make predictions using the loaded model
        predictions = model.predict(encoded_data)

        logging.info(f"predictions: {predictions}")

        logging.info("Prediction successful.")
        return {"prediction": predictions[0].tolist()}

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trigger_pipeline")
def trigger_pipeline(background_tasks: BackgroundTasks) -> dict:
    """
    Endpoint to trigger the ML pipeline.
    """
    def run_pipeline():
        script_path = "/app/trigger_ml_pipeline.sh"
        result = subprocess.run([script_path], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"Pipeline run successful: {result.stdout}")
        else:
            logging.error(f"Pipeline run failed: {result.stderr}")

    background_tasks.add_task(run_pipeline)
    return {"message": "Pipeline trigger initiated"}
