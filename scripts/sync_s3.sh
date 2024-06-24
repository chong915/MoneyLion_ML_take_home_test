SOURCE_BUCKET=s3://ml-source-bucket
# DVC_BUCKET=s3://ml-take-home-test-source-bucket
LOCAL_DATA_DIR=data/raw
DVC_REMOTE_NAME=myremote

# Ensure AWS CLI is installed
if ! command -v aws &> /dev/null
then
    echo "AWS CLI could not be found. Please install it and configure your credentials."
    exit
fi

# Ensure DVC is installed
if ! command -v dvc &> /dev/null
then
    echo "DVC could not be found. Please install it."
    exit
fi

# Create local data directory if it doesn't exist
mkdir -p $LOCAL_DATA_DIR

# Sync data from source S3 bucket to local directory
echo "Syncing data from $SOURCE_BUCKET to $LOCAL_DATA_DIR..."
aws s3 sync $SOURCE_BUCKET $LOCAL_DATA_DIR
