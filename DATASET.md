# Amelia-48 Dataset

AmeliaScenes is an automated method to select relevant ego-agents and produce diverse complex scene representations. It takes data generted by AmeliaSWIM and AmeliaMaps  to characterize relevant behaviours and interactions within airport operations, creatin a training, validation and testing splits.

## Overview

To run this repository, you first need to download the amelia dataset from [here](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/amelia-10.zip).

Make sure the dataset is structured as follows:

```bash
|-- datasets
    |-- amelia
        |-- assets
            | -- airport_icao
                | -- bkg_map.png
                | -- limits.json
                | -- bkg_map.pgw
                | -- airport_code_from_net.osm
        |-- graph_data_a10v01os
        |-- traj_data_a10v08

```
