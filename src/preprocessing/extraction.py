import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

def load_data(loan_path: str, payment_path: str, underwriting_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load loan, payment, and underwriting data from CSV files.

    Args:
    - loan_path (str): Path to the loan CSV file.
    - payment_path (str): Path to the payment CSV file.
    - underwriting_path (str): Path to the underwriting CSV file.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames for loan, payment, and underwriting data.
    """
    loan_df = pd.read_csv(loan_path, parse_dates=['applicationDate', 'originatedDate'])
    payment_df = pd.read_csv(payment_path)
    underwriting_df = pd.read_csv(underwriting_path)
    
    print(f'Loan df shape :{loan_df.shape}')
    print(f'Payment df shape :{payment_df.shape}')
    print(f'Underwriting df shape :{underwriting_df.shape}')
    
    return loan_df, payment_df, underwriting_df

def preprocess_data(loan_df: pd.DataFrame, underwriting_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the loan and underwriting data.

    Args:
    - loan_df (pd.DataFrame): Loan data.
    - underwriting_df (pd.DataFrame): Underwriting data.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.merge(loan_df, underwriting_df, left_on="clarityFraudId", right_on="underwritingid", how="left")
    df = df[df['isFunded'] == 1]
    
    # Drop irrelevant columns
    df = df[["loanId", "anon_ssn", "payFrequency", "apr", "applicationDate", "originatedDate", "nPaidOff", "loanStatus",
             "loanAmount", "originallyScheduledPaymentAmount", "state", "leadType", "leadCost", "fpStatus", "clearfraudscore"]]
    
    df = df.sort_values(by="applicationDate")
    
    # Convert columns to datetime format
    df['originatedDate'] = pd.to_datetime(df['originatedDate'], format='mixed')
    df['applicationDate'] = pd.to_datetime(df['applicationDate'], format='mixed')
    
    return df

def classify_loans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify loans into default and paid off categories and calculate application processing hours.

    Args:
    - df (pd.DataFrame): Preprocessed loan data.

    Returns:
    - pd.DataFrame: DataFrame with classified loans and processing hours.
    """
    default_classes = ['Internal Collection', 'External Collection', 'Returned Item', 
                       'Charged Off Paid Off', 'Settled Bankruptcy', 'Settlement Paid Off', 'Charged Off', 
                       'Settlement Paid Off']
    paid_off_classes = ['Paid Off Loan', "New Loan", "Pending Paid Off"]
    
    df.loc[df['loanStatus'].isin(default_classes).values, 'target'] = 1
    df.loc[df['loanStatus'].isin(paid_off_classes).values, 'target'] = 0
    df = df.loc[~df['target'].isnull()].reset_index(drop=True)
    
    # Calculate the difference in hours
    df['app_processing_hours'] = (df['originatedDate'] - df['applicationDate']).dt.total_seconds() / 3600
    
    return df

# def main():
#     loan_path = "./data/loan.csv"
#     payment_path = "./data/payment.csv"
#     underwriting_path = "./data/clarity_underwriting_variables.csv"
    
#     # Load data
#     loan_df, payment_df, underwriting_df = load_data(loan_path, payment_path, underwriting_path)
    
#     # Preprocess data
#     df = preprocess_data(loan_df, underwriting_df)
    
#     # Classify loans
#     df = classify_loans(df)
    
#     # Display the first few rows of the final DataFrame
#     print(df.head())

# if __name__ == "__main__":
#     main()
