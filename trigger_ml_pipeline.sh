#!/bin/bash

# Switch to the automated-pipeline branch
git checkout implement-dvc-versioning

# Pull the latest changes
git pull origin implement-dvc-versioning

# Set the PYTHONPATH to the current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Define variables
INPUT_FILE="./data/raw/loan.csv"
OUTPUT_DIR="./data/raw/synthetic_dataset"
LOAN_CSV_PATH="${OUTPUT_DIR}/loan.csv"

# Run the Python script to generate the synthetic dataset
python3 scripts/generate_synthetic_dataset.py --input_file $INPUT_FILE --output_dir $OUTPUT_DIR

# Run the Python script
python3 scripts/model_training.py --loan_csv_path $LOAN_CSV_PATH

# DVC commands to track the data and model directories
dvc add data/raw
dvc add data/processed
dvc add models

# Get the current date and time
current_date=$(date +"%Y-%m-%d %H:%M:%S")

# Commit the changes to Git
git add data/raw.dvc data/processed.dvc models.dvc .gitignore
git commit -m "Automated commit on ${current_date}: Track raw data, processed data, and models with DVC"

# Push the changes to the remote branch
git push origin implement-dvc-versioning

# Push the changes to the DVC remote
dvc push