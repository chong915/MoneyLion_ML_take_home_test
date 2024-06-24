import argparse
from src.preprocessing.extraction import load_data, preprocess_data, classify_loans, define_features
from src.preprocessing.data_encoder import DataEncoder
from src.training.hyperparams_optimization import hyperparameter_tuning
from src.training.train import train_model

import pandas as pd
import numpy as np
import os
from typing import Dict, Any

import dotenv

# Load environment variables from a .env file
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(loan_csv_path: str) -> None:
    """
    Main function to run the machine learning pipeline for training a loan risk model.

    Args:
    loan_csv_path (str): Path to the loan CSV file.
    """
    payment_csv_path = os.getenv('PAYMENT_CSV_PATH')
    underwriting_path = os.getenv('UNDERWRITING_CSV_PATH')

    # Load data
    loan_df, payment_df, underwriting_df = load_data(loan_csv_path, payment_csv_path, underwriting_path)

    # Preprocess data
    df = preprocess_data(loan_df, underwriting_df)

    # Classify loans
    df = classify_loans(df)

    num_feats, freq_feats, target_feats, predictor = define_features()

    selected_features = list(set(num_feats + freq_feats + target_feats + predictor))

    # Sort the data in chronological order to prevent data leakage
    df = df.sort_values(by="applicationDate")

    # Split data into train, validation, and test sets based on timeline
    train_size = int(len(df) * 0.8)  # 80% for training
    val_size = int(len(df) * 0.1)    # 10% for validation
    train_val_df = df[:train_size + val_size]
    test_df = df[train_size + val_size:]

    # Initialize the encoder
    encoder = DataEncoder(num_feats, target_feats, freq_feats)

    # Encode the training & validation data
    X_train_val = encoder.fit_transform(train_val_df, predictor)
    y_train_val = train_val_df[predictor]

    # Split the transformed data into train and validation splits
    X_train = X_train_val[:train_size]
    X_val = X_train_val[train_size:]

    y_train = y_train_val[:train_size]
    y_val = y_train_val[train_size:]

    # Do hyper-parameter tuning using validation data
    best_params = hyperparameter_tuning(X_train, X_val, y_train, y_val)

    # Once we find the best hyper-parameters, combine the train & validation sets and train again
    # then test it on the test set and give the metrics scores
    X_test = encoder.transform(test_df)
    y_test = test_df[predictor]

    dataset_dict = {
        'train_df': train_val_df,
        'y_train': train_val_df[predictor],
        'test_df': test_df,
        'y_test': y_test
    }

    scores_dict = train_model(dataset_dict, encoder, best_params, predictor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on loan data.")
    parser.add_argument('--loan_csv_path', type=str, required=True, help="Path to the loan CSV file.")

    args = parser.parse_args()

    main()
