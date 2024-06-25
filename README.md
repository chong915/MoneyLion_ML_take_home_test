# Loan Default Prediction

This project implements a machine learning pipeline and a FastAPI application for predicting loan defaults. The FastAPI app provides endpoints to predict loan default risk and trigger the ML pipeline on demand.

## Directory Structure

```plaintext
MoneyLion_ML_take_home_test/
├── .dvc/
│   └── .gitignore.py
├── app/
│   └── main.py
├── data/
│   ├── processed/
│   │   ├── test_df.joblib
│   │   └── train_df.joblib
│   ├── raw/
│   │   ├── synthetic_dataset/
│   │   │   └── loan.csv
│   │   ├── loan.csv
│   │   ├── payment.csv
│   │   └── clarity_underwriting_variables.csv
│   ├── .gitignore
│   ├── processed.dvc
|   └── raw.dvc
├── logs/
│   └── log_file.log
├── models/
│   ├── encoder.joblib
│   ├── lgb_model.joblib
│   └── metrics_json.joblib
├── notebook/
│   └── EDA.ipynb
├── prod_models/
│   ├── encoder.joblib
│   ├── lgb_model.joblib
│   └── metrics_json.joblib
├── scripts/
│   ├── run_training.sh
│   └── sync_s3.sh
├── src/
│   └── utils/
│       ├── data_preprocessing/
│       │   ├── __init__.py
│       │   ├── data_encoder.py
│       │   └── extraction.py
│       ├── model_training/
│       │   ├── __init__.py
│       │   ├── hyperparams_optimization.py
│       │   └── train.py
│       ├── compare_metrics.py
│       ├── generate_synthetic_dataset.py
│       └── model_training.py
├── .dockerignore
├── .dvcignore
├── .env
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── ml_crontab
├── models.dvc
├── prod_models.dvc
├── README.md
├── requirements.txt
├── startup.sh
└── trigger_ml_pipeline.sh
```

## Repository Overview

This repository contains the implementation for an automated end-to-end training pipeline for loan default prediction model. The project encompasses data preprocessing, model training, and deployment using FastAPI and Docker. The system utilizes DVC (Data Version Control) for managing data and model versioning.

The main components of the repository include:

- **Data Preprocessing**: Scripts for cleaning and preparing the input data.
- **Model Training**: Scripts for training machine learning models and hyperparameter tuning.
- **Deployment**: FastAPI application for serving the model and triggering the ML pipeline.
- **Docker Setup**: Configuration files for containerizing the application.
- **DVC**: Configuration for data versioning and model management.

## FastAPI Server

The FastAPI server provides an interface for interacting with the loan default prediction model. It exposes two main endpoints:

1. **Prediction Endpoint** (`/predict`):
   - **Method**: POST
   - **Description**: This endpoint accepts loan application data in JSON format and returns the predicted default risk.
   - **Input**: A JSON object containing the loan application details.
   - **Output**: A JSON object with the prediction result.

   **Example Request**:
   ```sh
   curl -X POST 'http://localhost:8000/predict' \
   -H 'Content-Type: application/json' \
   -d '{
       "loanId": 1,
       "anon_ssn": "123-45-6789",
       "payFrequency": "monthly",
       "apr": 5.5,
       "applicationDate": "2023-06-15",
       "originatedDate": "2023-06-16",
       "nPaidOff": 0,
       "loanStatus": "New Loan",
       "loanAmount": 10000.0,
       "originallyScheduledPaymentAmount": 300.0,
       "state": "CA",
       "leadType": "online",
       "leadCost": 50.0,
       "fpStatus": "active",
       "clearfraudscore": 1000.0
   }'
   ```
2. **Trigger Pipeline Endpoint** (`/trigger_pipeline`):
   - **Method**: POST
   - **Description**: This endpoint triggers the automated end-to-end machine learning pipeline which includes steps like data preprocessing, model training, and evaluation.
   - **Input**: A JSON object containing the loan application details.
   - **Output**: A JSON object indicating that the pipeline trigger has been initiated.

   **Example Request**:
   ```sh
   curl -X POST 'http://localhost:8000/trigger_pipeline'
   ```


## Project Components

1. **Data Preprocessing**:
   - Scripts located in `src/utils/data_preprocessing/` handle data extraction and encoding. These scripts prepare the raw input data for model training by cleaning and transforming it into a suitable format.

2. **Model Training**:
   - The `src/utils/model_training/` directory contains scripts for training machine learning models, optimizing hyperparameters, and evaluating model performance. The trained models and related metrics are saved for deployment.

3. **Deployment**:
   - The FastAPI server, defined in `app/main.py`, serves the trained model and provides endpoints for making predictions and triggering the ML pipeline. The server is containerized using Docker, making it easy to deploy and scale.

4. **Data Version Control (DVC)**:
   - DVC is used to manage datasets and model versions. This ensures reproducibility and allows easy tracking of changes over time. Configuration files for DVC are included in the repository.

5. **Docker**:
   - Dockerfiles and `docker-compose.yml` are provided to containerize the application. This allows for consistent deployment across different environments.

## How to Use

1. **Clone the Repository**:
   ```sh
   git clone git@github.com:chong915/MoneyLion_ML_take_home_test.git
   cd MoneyLion_ML_take_home_test
    ```

2. **Setup Environment Variables**:
    - Create a `.env` file in the parent directory with the following content:
    ```sh
    AWS_ACCESS_KEY_ID=your_aws_access_key_id
    AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
    AWS_PROFILE=your_aws_profile
    S3_BUCKET_LINK=your_s3_bucket_link
    GITHUB_EMAIL=your_github_email
    ```

3. **Setup AWS S3 Bucket**:
    - Configure the AWS CLI and setup DVC to use your S3 bucket:
    ```sh
    aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
    aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
    aws configure set default.region ap-southeast-2

    # DVC setup
    dvc remote add -d myremote $S3_BUCKET_LINK
    dvc remote modify myremote access_key_id $AWS_ACCESS_KEY_ID
    dvc remote modify myremote secret_access_key $AWS_SECRET_ACCESS_KEY
    ```

4. **Generate SSH Keys for Git**:
    - Generate SSH keys to allow DVC to interact with Git:
    ```sh
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
    ssh-keyscan github.com >> /root/.ssh/known_hosts
    ```
    - Add the generated SSH key to your Git hosting service (e.g., Github)
    ```sh
    cat /root/.ssh/id_rsa.pub
    ```

5. **Build and Run Docker Container**:
    ```sh
    docker-compose up --build
    ```

6. **Interact with the FastAPI Server**：
    - Use the provided `curl` commands to interact with the prediction and pipeline endpoints.
    - Please refer to the FastAPI section above for example requests.