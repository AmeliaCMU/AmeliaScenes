# Amelia-48 Dataset

AmeliaScenes is an automated method to select relevant ego-agents and produce diverse complex scene representations. It takes data generted by AmeliaSWIM and AmeliaMaps  to characterize relevant behaviours and interactions within airport operations, creatin a training, validation and testing splits.

## Overview

To run this repository, you first need to download the amelia dataset. Follow the instructions [here]().

Make sure the dataset is structured as follows: 

```bash
|-- datasets
    |-- amelia
        |-- assets
            | -- airport_icao
                | -- bkg_map.png
        |-- graph_data_a10v01os
        |-- traj_data_a10v08

```