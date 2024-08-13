#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create --name $1 python=3.9 -y
conda activate $1


mkdir $1
cd $1

# array of arguments
repos=(git@github.com:AmeliaCMU/AmeliaScenes.git git@github.com:AmeliaCMU/AmeliaMaps.git git@github.com:AmeliaCMU/AmeliaTF.git git@github.com:AmeliaCMU/AmeliaInference.git git@github.com:AmeliaCMU/AmeliaDataTools.git git@github.com:AmeliaCMU/AmeliaSWIM.git)
amelia_framework=(AmeliaScenes AmeliaMaps AmeliaTF AmeliaInference AmeliaDataTools AmeliaSWIM)

# loop through the array
for i in "${!repos[@]}"; do
    echo "Cloning ${repos[$i]}"
    git clone ${repos[$i]}
    pip install -e ${amelia_framework[$i]}
done