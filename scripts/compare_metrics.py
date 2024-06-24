import joblib
import os
import argparse
import shutil
from typing import Tuple, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_metrics(file_path: str) -> Optional[Dict]:
    """
    Load metrics from a joblib file.

    Args:
    - file_path (str): Path to the metrics file.

    Returns:
    - Optional[Dict]: Loaded metrics dictionary or None if the file does not exist.
    """
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None


def compare_f1_scores(current_metrics_path: str, deployed_metrics_path: str) -> Tuple[float, float]:
    """
    Compare the f1_scores from the current and deployed metrics files.

    Args:
    - current_metrics_path (str): Path to the current metrics file.
    - deployed_metrics_path (str): Path to the deployed metrics file.

    Returns:
    - Tuple[float, float]: A tuple containing the current and deployed f1_scores.
    """
    current_metrics = load_metrics(current_metrics_path)
    deployed_metrics = load_metrics(deployed_metrics_path)

    current_f1_score = current_metrics['test']['f1_score'] if current_metrics else 0
    deployed_f1_score = deployed_metrics['test']['f1_score'] if deployed_metrics else 0

    return current_f1_score, deployed_f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare f1_scores from current and deployed metrics.")
    parser.add_argument('--current_metrics', type=str, required=True, help="Path to the current metrics file.")
    parser.add_argument('--deployed_metrics', type=str, required=True, help="Path to the deployed metrics file.")
    parser.add_argument('--deploy_dir', type=str, required=True, help="Directory to deploy the new model if it's better.")

    args = parser.parse_args()

    current_f1_score, deployed_f1_score = compare_f1_scores(args.current_metrics, args.deployed_metrics)

    logging.info(f"Deployed f1_score: {deployed_f1_score}")
    logging.info(f"Current f1_score: {current_f1_score}")

    if current_f1_score > deployed_f1_score:
        logging.info("Current model is better. Deploying the new model.")
        # Clear the deploy directory
        if os.path.exists(args.deploy_dir):
            shutil.rmtree(args.deploy_dir)
        os.makedirs(args.deploy_dir)

        # Move the models to the deployment directory
        for item in os.listdir('models'):
            s = os.path.join('models', item)
            d = os.path.join(args.deploy_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
                
        logging.info("Model deployment successful.")
        exit(0)  # Indicate successful deployment
    else:
        logging.info("Deployed model is better. Keeping the deployed model unchanged.")
        exit(1)  # Indicate no deployment
