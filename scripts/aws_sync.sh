#!/bin/bash

# Usage: ./upload_to_s3.sh <local_folder> <s3_bucket_name> [<s3_folder>]

# LOCAL_FOLDER="$1"
# S3_BUCKET="$2"
# S3_FOLDER="$3"


LOCAL_FOLDER="/Volumes/acanvil/datasets/amelia/traj_data_a42v99percentile/proc_full_scenes"
S3_BUCKET="cmu-amelia"
S3_FOLDER="Amelia42-Mini/traj_data_a42v99percentile/proc_full_scenes"

if [ -z "$LOCAL_FOLDER" ] || [ -z "$S3_BUCKET" ]; then
    echo "Usage: $0 <local_folder> <s3_bucket_name> [<s3_folder>]"
    exit 1
fi

if [ -n "$S3_FOLDER" ]; then
    DEST="s3://$S3_BUCKET/$S3_FOLDER/"
else
    DEST="s3://$S3_BUCKET/"
fi

aws s3 sync "$LOCAL_FOLDER" "$DEST" --exclude "*.DS_Store"

if [ $? -eq 0 ]; then
    echo "Upload completed successfully."
else
    echo "Upload failed."
    exit 2
fi
