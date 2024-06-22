from src.preprocessing.extraction import load_data, preprocess_data, classify_loans
from src.preprocessing.data_cleaning import define_features

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder, CountEncoder

from typing import List


class DataEncoder:
    def __init__(self, num_feats: List[str], target_feats: List[str], freq_feats: List[str]):
        self.num_feats = num_feats
        self.target_feats = target_feats
        self.freq_feats = freq_feats
        self.target_encoder = TargetEncoder(cols=target_feats)
        self.freq_encoder = CountEncoder(cols=freq_feats)

    def fit_transform(self, train_df: pd.DataFrame, target: str):
        # Encode target features
        train_target_encoded = self.target_encoder.fit_transform(train_df[self.target_feats], train_df[target])
        train_target_encoded.columns = [f"{col}_target" for col in self.target_feats]
        
        # Encode frequency features
        train_freq_encoded = self.freq_encoder.fit_transform(train_df[self.freq_feats])
        train_freq_encoded.columns = [f"{col}_freq" for col in self.freq_feats]
        
        # Select numerical features
        train_num_feats = train_df[self.num_feats]
        
        # Combine all features
        train_encoded = pd.concat([train_target_encoded, train_freq_encoded, train_num_feats], axis=1)
        
        return train_encoded

    def transform(self, test_df: pd.DataFrame):
        # Encode target features
        test_target_encoded = self.target_encoder.transform(test_df[self.target_feats])
        test_target_encoded.columns = [f"{col}_target" for col in self.target_feats]
        
        # Encode frequency features
        test_freq_encoded = self.freq_encoder.transform(test_df[self.freq_feats])
        test_freq_encoded.columns = [f"{col}_freq" for col in self.freq_feats]
        
        # Select numerical features
        test_num_feats = test_df[self.num_feats]
        
        # Combine all features
        test_encoded = pd.concat([test_target_encoded, test_freq_encoded, test_num_feats], axis=1)
        
        return test_encoded