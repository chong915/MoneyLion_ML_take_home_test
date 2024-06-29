import os
import random
import logging
import argparse
from datetime import timedelta

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(input_file: str, output_dir: str) -> None:
    """
    Generate a synthetic dataset from the provided loan.csv file by selecting a random time window.

    Args:
    - input_file (str): Path to the input loan.csv file.
    - output_dir (str): Directory to save the synthetic dataset.
    """
    logging.info(f"Loading data from {input_file}")

    # Load the dataset
    df = pd.read_csv(input_file)

    # Convert applicationDate to datetime
    df['applicationDate'] = pd.to_datetime(df['applicationDate'], format='mixed')

    # Sort the dataframe by applicationDate
    df_sorted = df.sort_values(by='applicationDate')

    # Get the minimum and maximum applicationDate
    min_date = df_sorted['applicationDate'].min()
    max_date = df_sorted['applicationDate'].max()

    # Calculate the total date range
    total_days = (max_date - min_date).days

    # Use min_date as start_date
    start_date = min_date

    # Randomly select a time window (e.g., between 30 and max_days)
    time_window_days = random.randint(300, total_days)
    end_date = start_date + timedelta(days=time_window_days)

    # Create a subset of the dataframe within the time window
    synthetic_df = df_sorted[(df_sorted['applicationDate'] >= start_date) & (df_sorted['applicationDate'] <= end_date)]

    # Create a directory to save the synthetic dataset
    os.makedirs(output_dir, exist_ok=True)

    # Save the synthetic subset dataset to a CSV file
    synthetic_file_path = f'{output_dir}/loan.csv'
    synthetic_df.to_csv(synthetic_file_path, index=False)

    logging.info(f"Synthetic dataset saved to {synthetic_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a synthetic dataset from loan.csv')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input loan.csv file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the synthetic dataset')

    args = parser.parse_args()
    main(args.input_file, args.output_dir)
