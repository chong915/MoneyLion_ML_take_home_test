import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

import pandas as pd
import numpy as np

import lightgbm as lgb

# Hardcoded paths
MODEL_PATH = './models/lgb_model.joblib'
ENCODER_PATH = './models/encoder.joblib'
METRICS_JSON_PATH = './models/metrics_json.joblib'

TRAIN_DF_PATH = './data/processed/train_df.joblib'
TEST_DF_PATH = './data/processed/test_df.joblib'

import os

def train_model(dataset_dict, encoder, best_params, predictor):
    train_df = dataset_dict['train_df']
    test_df = dataset_dict['test_df']

    X_train = encoder.fit_transform(train_df, predictor)
    X_test = encoder.fit_transform(test_df, predictor)
    y_train = dataset_dict['y_train']
    y_test = dataset_dict['y_test']

    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['verbosity'] = -1
    best_params['num_leaves'] = int(best_params['num_leaves'])

    print(f"Current working directory: {os.getcwd()}")
    print(f"X_train shape : {X_train.shape}")
    print(f"y_train shape : {y_train.shape}")

    dtrain_full = lgb.Dataset(X_train, label=y_train)

    # Train final model on combined train and validation sets
    final_model = lgb.train(best_params, train_set=dtrain_full, num_boost_round=best_params['n_estimators'])

    # Save the final model and the encoder object
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    # Save train_df and test_df using joblib
    joblib.dump(train_df, TRAIN_DF_PATH)
    joblib.dump(test_df, TEST_DF_PATH)


    # Load the final model and the encoder object
    loaded_model = joblib.load(MODEL_PATH)
    loaded_encoder = joblib.load(ENCODER_PATH)

    # Load the saved dataframes
    saved_train_df = joblib.load(TRAIN_DF_PATH)
    saved_test_df = joblib.load(TEST_DF_PATH)

    # Inference on test data
    X_test_encoded = loaded_encoder.transform(saved_test_df)
    predictions = loaded_model.predict(X_test_encoded)
    print(f"Predictions : {predictions}")
    binary_predictions = np.round(predictions)

    # Evaluate the model on the test set
    test_f1_score = f1_score(y_test, binary_predictions)
    test_precision = precision_score(y_test, binary_predictions)
    test_recall = recall_score(y_test, binary_predictions)
    test_accuracy = accuracy_score(y_test, binary_predictions)

    # Evaluate the model on the training set
    train_predictions = loaded_model.predict(X_train)
    train_binary_predictions = np.round(train_predictions)

    train_f1_score = f1_score(y_train, train_binary_predictions)
    train_precision = precision_score(y_train, train_binary_predictions)
    train_recall = recall_score(y_train, train_binary_predictions)
    train_accuracy = accuracy_score(y_train, train_binary_predictions)

    scores_dict = {
        'train': {
            'accuracy': round(train_accuracy, 4),
            'precision': round(float(train_precision), 4),
            'recall': round(float(train_recall), 4),
            'f1_score': round(float(train_f1_score), 4)
        },
        'test': {
            'accuracy': round(test_accuracy, 4),
            'precision': round(float(test_precision), 4),
            'recall': round(float(test_recall), 4),
            'f1_score': round(float(test_f1_score), 4)
        }
    }

    joblib.dump(scores_dict, METRICS_JSON_PATH)
    # print("Final Training Scores:")
    # print(f"Accuracy: {train_accuracy:.4f}")
    # print(f"Precision: {train_precision:.4f}")
    # print(f"Recall: {train_recall:.4f}")
    # print(f"F1 Score: {train_f1_score:.4f}")

    # print("\nTest Scores:")
    # print(f"Accuracy: {test_accuracy:.4f}")
    # print(f"Precision: {test_precision:.4f}")
    # print(f"Recall: {test_recall:.4f}")
    # print(f"F1 Score: {test_f1_score:.4f}")

    # # Detailed classification report
    # print("\nClassification Report on Test Data:")
    # print(classification_report(y_test, binary_predictions))

    return scores_dict