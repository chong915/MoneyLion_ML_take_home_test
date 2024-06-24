from category_encoders import TargetEncoder, CountEncoder
import pandas as pd

from typing import List


class DataEncoder:
    """
    A class used to encode and transform features for dataset.

    Attributes:
    ----------
    num_feats : List[str]
        A list of numerical feature names.
    target_feats : List[str]
        A list of categorical feature names to be target encoded.
    freq_feats : List[str]
        A list of categorical feature names to be frequency encoded.
    target_encoder : TargetEncoder
        Encoder for target features.
    freq_encoder : CountEncoder
        Encoder for frequency features.
    """

    def __init__(self, num_feats: List[str], target_feats: List[str], freq_feats: List[str]):
        """
        Initialize the DataEncoder with the given feature lists.

        Parameters:
        ----------
        num_feats : List[str]
            A list of numerical feature names.
        target_feats : List[str]
            A list of categorical feature names to be target encoded.
        freq_feats : List[str]
            A list of categorical feature names to be frequency encoded.
        """
        self.num_feats = num_feats
        self.target_feats = target_feats
        self.freq_feats = freq_feats
        self.target_encoder = TargetEncoder(cols=target_feats)
        self.freq_encoder = CountEncoder(cols=freq_feats)


    def feature_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset by converting date columns to datetime and calculating the application processing hours.

        Parameters:
        ----------
        df : pd.DataFrame
            The dataframe to transform.

        Returns:
        -------
        pd.DataFrame
            The transformed dataframe with additional 'app_processing_hours' column.
        """
        # Convert columns to datetime format
        df['originatedDate'] = pd.to_datetime(df['originatedDate'], format='mixed')
        df['applicationDate'] = pd.to_datetime(df['applicationDate'], format='mixed')

        # Calculate the difference in hours
        df['app_processing_hours'] = (df['originatedDate'] - df['applicationDate']).dt.total_seconds() / 3600

        return df
        

    def fit_transform(self, train_df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Fit and transform the training dataframe using target and frequency encoders.

        Parameters:
        ----------
        train_df : pd.DataFrame
            The training dataframe.
        target : str
            The target column name.

        Returns:
        -------
        pd.DataFrame
            The encoded and transformed training dataframe.
        """
        # Feature engineering
        train_df = self.feature_transform(train_df)

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


    def transform(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the test dataframe using the fitted target and frequency encoders.

        Parameters:
        ----------
        test_df : pd.DataFrame
            The test dataframe.

        Returns:
        -------
        pd.DataFrame
            The encoded and transformed test dataframe.
        """
        # Feature engineering
        test_df = self.feature_transform(test_df)

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