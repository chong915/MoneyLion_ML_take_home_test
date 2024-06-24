#!/bin/bash

# Define variables
FOLDER_DIR="/Users/munchong/Desktop/MoneyLion/MoneyLion_ML_take_home_test"
BRANCH_NAME="implement-dvc-versioning"

INPUT_FILE="${FOLDER_DIR}/data/raw/loan.csv"
OUTPUT_DIR="${FOLDER_DIR}/data/raw/synthetic_dataset"
LOAN_CSV_PATH="${OUTPUT_DIR}/loan.csv"
CURRENT_METRICS="${FOLDER_DIR}/models/metrics_json.joblib"

DEPLOY_DIR="${FOLDER_DIR}/prod_models"
DEPLOYED_METRICS="${DEPLOY_DIR}/metrics_json.joblib"

LOG_FILE="${FOLDER_DIR}/logs/log_file.log"
LOG_DIR=$(dirname "$LOG_FILE")


# Navigate to the project directory
cd $FOLDER_DIR || exit

# Activate the virtual environment
source ${FOLDER_DIR}/env/bin/activate

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Log the current directory
echo "Current directory: $(pwd)" >> $LOG_FILE

# Switch to the target branch and pull the latest changes
git checkout $BRANCH_NAME

# Pull the latest changes
git pull origin $BRANCH_NAME

# Log after git pull
echo "Git pull completed at $(date)" >> $LOG_FILE

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the Python script to generate the synthetic dataset
python3 ${FOLDER_DIR}/src/generate_synthetic_dataset.py --input_file $INPUT_FILE --output_dir $OUTPUT_DIR

echo "Finished generating synthetic dataset" >> $LOG_FILE

# Run the Python script
python3 ${FOLDER_DIR}/src/model_training.py --loan_csv_path $LOAN_CSV_PATH

echo "Finished training model" >> $LOG_FILE

# DVC commands to track the data and model directories
dvc add ${FOLDER_DIR}/data/raw >> $LOG_FILE 2>&1
dvc add ${FOLDER_DIR}/data/processed >> $LOG_FILE 2>&1
dvc add ${FOLDER_DIR}/models >> $LOG_FILE 2>&1

# Get the current date and time
current_date=$(date +"%Y-%m-%d %H:%M:%S")

# Commit the changes to Git
git add ${FOLDER_DIR}/data/raw.dvc ${FOLDER_DIR}/data/processed.dvc ${FOLDER_DIR}/models.dvc ${FOLDER_DIR}/.gitignore ${FOLDER_DIR}/logs/log_file.log
git commit -m "Automated commit on ${current_date}: Track raw data, processed data, and models with DVC"

# Push the changes to the remote branch
git push origin $BRANCH_NAME >> $LOG_FILE 2>&1

# Push the changes to the DVC remote
dvc push >> $LOG_FILE 2>&1

# Pull the deployed model's directory from DVC if it exists
if dvc pull $DEPLOY_DIR; then
    echo "Deployed models pulled successfully"
else
    echo "No deployed models found, skipping pull"
fi

# Compare the current model's test f1_score with the deployed one
python3 ${FOLDER_DIR}/src/compare_metrics.py --current_metrics $CURRENT_METRICS --deployed_metrics $DEPLOYED_METRICS --deploy_dir $DEPLOY_DIR

# Capture the exit status of the compare_metrics.py script
status=$?
echo "compare_metrics.py exit status: $status" >> $LOG_FILE

# Check the exit status of the compare_metrics.py script
if [ $status -eq 0 ]; then
    echo "Deploying current model to prod" >> $LOG_FILE
    # Track and push the deployed model directory if new model is deployed
    dvc add $DEPLOY_DIR >> $LOG_FILE 2>&1
    git add $DEPLOY_DIR.dvc ${FOLDER_DIR}.gitignore
    f1_score=$(python3 -c "import joblib; metrics = joblib.load('$DEPLOYED_METRICS'); print(metrics['test']['f1_score'])")
    echo "The new test f1-score :${f1_score}" >> $LOG_FILE
    git commit -m "Deploying new model with f1_score: ${f1_score}"
    git push origin $BRANCH_NAME >> $LOG_FILE 2>&1
    dvc push >> $LOG_FILE 2>&1
fi

# Log the end time
echo "Script ended at $(date)" >> $LOG_FILE