import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


def split_data(df: pd.DataFrame, test_size: float = 0.2, validation_size: float = 0.25):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=validation_size, random_state=42)
    return train_df, val_df, test_df


def define_features():
    # Example usage
    num_feats = ['apr', 'loanAmount', 'originallyScheduledPaymentAmount', 'leadCost', 'app_processing_hours', 'clearfraudscore']
    freq_feats = ['payFrequency', 'nPaidOff', 'state', 'fpStatus']
    target_feats = ['payFrequency', 'nPaidOff', 'state', 'fpStatus']
    predictor = ['target']

    return num_feats, freq_feats, target_feats, predictor