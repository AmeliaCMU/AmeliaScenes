#!/bin/bash
set -e
# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Please provide the environment name as an argument."
  echo "Usage: $0 <env_name>"
  exit 1
fi

ENV_NAME=$1
PYTHON_VERSION="3.9"
# Create the conda environment with Python 3.9
eval "$(conda shell.bash hook)"
echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."

conda create -n $ENV_NAME python=$PYTHON_VERSION -y
conda activate $ENV_NAME

mkdir $1
cd $1

repos=(git@github.com:AmeliaCMU/AmeliaScenes.git git@github.com:AmeliaCMU/AmeliaMaps.git git@github.com:AmeliaCMU/AmeliaTF.git git@github.com:AmeliaCMU/AmeliaInference.git git@github.com:AmeliaCMU/AmeliaDataTools.git git@github.com:AmeliaCMU/AmeliaSWIM.git)
amelia_framework=(AmeliaScenes AmeliaMaps AmeliaTF AmeliaInference AmeliaDataTools AmeliaSWIM)

# loop through the array
for i in "${!repos[@]}"; do
    echo "Cloning ${repos[$i]}"
    git clone ${repos[$i]} || { echo "Failed to clone ${repos[$i]}. Please make sure you have SSH access or use the SSH method for cloning."; exit 1; }
    pip install -e ${amelia_framework[$i]}
done