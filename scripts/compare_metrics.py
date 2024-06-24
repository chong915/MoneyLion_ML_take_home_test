import joblib
import os
import argparse
import shutil

def load_metrics(file_path):
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None

def compare_f1_scores(current_metrics_path, deployed_metrics_path):
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

    print(f"Deployed f1_score: {deployed_f1_score}")
    print(f"Current f1_score: {current_f1_score}")

    if current_f1_score > deployed_f1_score:
        print("Current model is better. Deploying the new model.")
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
        exit(0)  # Indicate successful deployment
    else:
        print("Deployed model is better. Keeping the deployed model unchanged.")
        exit(1)  # Indicate no deployment
