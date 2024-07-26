#!/bin/bash

# conda create --name $1 python=3.9
# conda activate $1


mkdir amelia
cd amelia

# array of arguments
repos=(git@github.com:AmeliaCMU/AmeliaScenes.git git@github.com:AmeliaCMU/AmeliaMaps.git git@github.com:AmeliaCMU/AmeliaTF.git git@github.com:AmeliaCMU/AmeliaInference.git)
amelia_framework=(AmeliaScenes AmeliaMaps AmeliaTF AmeliaInference)

# loop through the array
for i in "${!repos[@]}"; do
    echo "Cloning ${repos[$i]}"
    git clone ${repos[$i]}
    pip install -e ${amelia_framework[$i]}
done