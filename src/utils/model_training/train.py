import os
import logging
import joblib
import dotenv

from typing import Dict

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from src.utils.data_preprocessing import DataEncoder

# Load environment variables from a .env file
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_model(dataset_dict: Dict[str, pd.DataFrame], encoder: DataEncoder, best_params: Dict, predictor: str) -> Dict[str, Dict[str, float]]:
    """
    Train the LightGBM model, save the model and related objects, and evaluate its performance.

    Parameters:
    ----------
    dataset_dict : dict
        A dictionary containing the training and test datasets along with their corresponding target values. 
        Keys should include 'train_df', 'test_df', 'y_train', and 'y_test'.
    encoder : DataEncoder
        An instance of the DataEncoder class used for encoding the data.
    best_params : dict
        A dictionary of the best hyperparameters obtained from hyperparameter tuning.
    predictor : str
        The name of the target variable.

    Returns:
    -------
    dict
        A dictionary containing the evaluation scores for both training and test sets.

    Description:
    ------------
    This function trains a LightGBM model using the provided training data and best hyperparameters.
    It saves the trained model, encoder, and datasets using joblib. It then loads the saved objects and evaluates
    the model's performance on the test set. The evaluation metrics include accuracy, precision, recall, and F1 score,
    which are logged and returned in a dictionary.

    Example:
    --------
    >>> scores = train_model(dataset_dict, encoder, best_params, 'target')
    >>> print(scores)
    """
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

    logging.info(f"X_train shape : {X_train.shape}")
    logging.info(f"y_train shape : {y_train.shape}")

    dtrain_full = lgb.Dataset(X_train, label=y_train)

    # Train final model on combined train and validation sets
    final_model = lgb.train(best_params, train_set=dtrain_full, num_boost_round=best_params['best_iteration'])

    MODEL_PATH = os.getenv('MODEL_PATH')
    ENCODER_PATH = os.getenv('ENCODER_PATH')
    TRAIN_DF_PATH = os.getenv('TRAIN_DF_PATH')
    TEST_DF_PATH = os.getenv('TEST_DF_PATH')

    # Check the attributes of the loaded encoder
    print("Numerical Features:", encoder.num_feats)
    print("Target Encoded Features:", encoder.target_feats)
    print("Frequency Encoded Features:", encoder.freq_feats)
    print("Target Encoder:", encoder.target_encoder)
    print("Frequency Encoder:", encoder.freq_encoder)


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

    joblib.dump(scores_dict, os.getenv('METRICS_JSON_PATH'))
    logging.info("--------------------------------\n\nFinal Training Scores:")
    logging.info(f"Accuracy: {train_accuracy:.4f}")
    logging.info(f"Precision: {train_precision:.4f}")
    logging.info(f"Recall: {train_recall:.4f}")
    logging.info(f"F1 Score: {train_f1_score:.4f}")

    logging.info("--------------------------------\n\nTest Scores:")
    logging.info(f"Accuracy: {test_accuracy:.4f}")
    logging.info(f"Precision: {test_precision:.4f}")
    logging.info(f"Recall: {test_recall:.4f}")
    logging.info(f"F1 Score: {test_f1_score:.4f}")

    return scores_dict