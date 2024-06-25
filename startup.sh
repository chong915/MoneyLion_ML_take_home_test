#!/bin/sh

# Configure Github
git config --global user.email $GITHUB_EMAIL
git config --global user.name $GITHUB_NAME

# Configure AWS CLI
aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set default.region ap-southeast-2

# DVC setup
dvc remote add -d myremote $S3_BUCKET_LINK
dvc remote modify myremote access_key_id $AWS_ACCESS_KEY_ID
dvc remote modify myremote secret_access_key $AWS_SECRET_ACCESS_KEY

# Pull DVC data
dvc pull prod_models.dvc

# Run the application
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
