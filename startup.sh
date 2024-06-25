#!/bin/sh

# # Set permissions for the SSH keys
# chmod 600 /root/.ssh/id_rsa /root/.ssh/id_rsa.pub

# Add GitHub to known hosts to avoid SSH prompt
# ssh-keyscan github.com >> /root/.ssh/known_hosts

# # Ensure AWS credentials are available
# export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
# export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
# export AWS_PROFILE=${AWS_PROFILE}
# export GITHUB_EMAIL=${GITHUB_EMAIL}
# export GITHUB_NAME=${GITHUB_NAME}

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
