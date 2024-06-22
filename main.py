import argparse
from src.preprocessing.extraction import load_data, preprocess_data, classify_loans
from src.preprocessing.data_cleaning import define_features
from src.preprocessing.data_encoder import DataEncoder

from src.training.hyperparams_optimization import hyperparamter_tuning
from src.training.train import train_model

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def main(loan_path, payment_path, underwriting_path):
    # Load data
    loan_df, payment_df, underwriting_df = load_data(loan_path, payment_path, underwriting_path)

    # Preprocess data
    df = preprocess_data(loan_df, underwriting_df)

    # Classify loans
    df = classify_loans(df)

    num_feats, freq_feats, target_feats, predictor = define_features()

    selected_features = list(set(num_feats + freq_feats + target_feats + predictor))

    # Split data into train and test sets
    df = df[selected_features]
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Initialize the encoder
    encoder = DataEncoder(num_feats, target_feats, freq_feats)

    # Encode the training data
    X_train = encoder.fit_transform(train_df, predictor)
    y_train = train_df[predictor]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    best_params = hyperparamter_tuning(X_train, X_val, y_train, y_val)

    X_test = encoder.transform(test_df)
    y_test = test_df[predictor]

    dataset_dict = {
        'train_df': train_df,
        'y_train': train_df[predictor],
        'test_df': test_df,
        'y_test': y_test
    }

    scores_dict = train_model(dataset_dict, encoder, best_params, predictor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on loan data.")
    parser.add_argument('--loan_path', type=str, required=True, help="Path to the loan CSV file.")
    parser.add_argument('--payment_path', type=str, required=True, help="Path to the payment CSV file.")
    parser.add_argument('--underwriting_path', type=str, required=True, help="Path to the underwriting CSV file.")

    args = parser.parse_args()

    main(args.loan_path, args.payment_path, args.underwriting_path)
