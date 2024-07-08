# AmeliaScenes

## Overview

AmeliaScenes is a tool for generating airport surface movement scenes from raw trajectory data collected with [AmeliaSWIM](https://github.com/AmeliaCMU/AmeliaSWIM). It takes CSV files containing various fields representing agent trajectory data and produces scenes based on certain specifications, including number of desired agents and scene length. 

<p align="center">
</p>

<div align="center">
  <img width="800" src="./assets/kbos.gif">
  <h5>Scene Example from Boston Logan International Airport (KBOS).</h5>
</div>

AmeliaScenes also provides scene and per-agent characterization tools as meta information for each agent's kinematic and interactive profile. 

<div align="center">
  <img width="800" src="./assets/scoring.png">
  <h5>Scene Scoring Example. </h5>
</div>

Finally, AmeliaScenes also provides a dataset splitting script with various ```train/val/test``` splitting strategies. 

## Pre-requisites

### Dataset 

To run this repository, you first need to download the amelia dataset. Follow the instructions [here](https://github.com/AmeliaCMU/AmeliaScenes/DATASET.md) to download and setup the dataset. 

Once downloaded, create a symbolic link into  ```datasets```:
```
cd datasets
ln -s /path/to/the/amelia/dataset . 
```

### Installation

This repository can be installed following the instructions [here](https://github.com/AmeliaCMU/AmeliaScenes/INSTALL.md). However, we recommend to setup all of our Amelia Framework tools. You can do so following the instructions [here](https://github.com/AmeliaCMU/AmeliaScenes/INSTALL.md)


## How to use

Activate your amelia environment (**Please follow the installation instructions above**):
```bash
conda activate amelia
```

### Generating scenes from raw data

Once you've installed the tools, and created the amelia enviroment. Run:
```bash
cd amelia_scenes/scene_processing
python run_processor.py --airport [airport_icao] --parallel
```

Where:
- `[airport_icao]`: [ICAO](https://en.wikipedia.org/wiki/ICAO_airport_code) code of the airport to be processed. It can be one of the following: `kbos`, `kdca`, `kewr`, `kjfk`, `klax`, `kmdw`, `kmsy`, `ksea`, `ksfo`, `panc`. By default it is set to `kbos`.
- `[parallel]`: If the processing should be done in parallel. By default it is set to `True`.

Additional parameters can also be specified:
```bash
cd amelia_scenes/scene_processing
python run_processor.py --airport [airport_icao] --to_process [scenes | metas | both] --parallel \
                        --to_process [to_process] \
                        --base_dir [path_to_dataset] \ 
                        --traj_version [version] 
                        --graph_version [version] \
                        --parallel [parallel] \
                        --overwite [overwite] \
                        --perc_process [percentage] \
                        --seed [seed]
```
Where:
- `[to_process]`: What to process. By default is set to `both`. Possible options are:
    - `scenes`: only generate scenes from the raw files
    - `metas`: generates meta information from already generated scenes. It uses scene scoring tools. 
    - `both`: generates scenes and meta information, simultaneously. 
- `[base_dir]`: Path to the dataset. By defaulult the path is set to `../../datasets/amelia`.
- `[traj_version]`: Version of the trajectory data. By default it is set to `a10v08`.
- `[graph_version]`: Version of the graph data. By default it is set to `a10v01os`.
- `[overwrite]`: If the processing should overwrite the existing data. By default it is set to `True`.
- `[perc_process]`: Top limit vizualization of the data being processed. By default it is set to `1.0`.
- `[seed]`: Seed for the random number generator. By default it is set to `42`.

#### Expected output

The scene processor should generate scene files for a given CSV file into ```datasets/amelia/traj_data_{version}/proc_trajectories/{airport_icao}/{raw_file_tagname}```. Each scene file is a pickle file following the format ```scene_id.pkl```. 

For example if the input file is ```KBOS_1_1672531200.csv```, found in:
```bash
|-- datasets
    |-- amelia
        |-- traj_data_a10v08
            | -- raw_trajectories
                | -- kbos
                    | -- KBOS_1_1672531200.csv
                | -- other airports
```

The output scenes will be in:
 ```bash
|-- datasets
    |-- amelia
        |-- traj_data_a10v08
            | -- raw_trajectories
            | -- proc_scenes
                | -- kbos
                    | -- KBOS_1_1672531200
                        | -- 00000.pkl
                        ...
                        | -- xxxxx.pkl
```

<hr>

### Characterizing scenes 

**TODO**


<hr>

### Creating dataset splits

**TODO**
``` bash
python amelia_scenes/scene_splitting/create_splits.py --split_type [split_type] --base_dir [path_to_dataset] --traj_version [version] --graph_version [version] --seed [seed]
```

# BibTeX

If you find our work useful in your research, please cite us!

```
@article{navarro2024amelia,
  title={Amelia: A Large Model and Dataset for Airport Surface
Movement Forecasting},
  author={Navarro, Ingrid and Ortega-Kral, Pablo and Patrikar, Jay, and Haichuan, Wang and Park, Jong Hoon and Oh, Jean and Scherer, Sebastian},
  journal={arXiv preprint arXiv:2309.08889},
  year={2024}
}
```